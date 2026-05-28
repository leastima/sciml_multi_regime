import os, sys, time
import numpy as np
import argparse
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import wandb
import matplotlib.pyplot as plt
from datetime import datetime
import logging
from utils import logging_utils
logging_utils.config_logger()
from utils.YParams import YParams
from utils.data_utils import get_data_loader
from utils.optimizer_utils import set_scheduler, set_optimizer
from utils.loss_utils import LossMSE
from utils.misc_utils import compute_grad_norm, vis_fields, l2_err
from utils.domains import DomainXY
from utils.sweeps import sweep_name_suffix
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap as ruamelDict
from collections import OrderedDict

# models
import models.ffn
import models.fno

def print_mem():
    print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
    print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
    print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))

def set_seed(params, world_size):
    seed = params.seed
    if seed is None:
        seed = np.random.randint(10000)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if world_size > 0:
        torch.cuda.manual_seed_all(seed)

def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1024/1024

class Trainer():
    """ trainer class """
    def __init__(self, params, args):
        self.sweep_id = args.sweep_id
        self.root_dir = args.root_dir
        self.config = args.config 
        self.run_num = args.run_num
        self.world_size = 1
        
        if 'WORLD_SIZE' in os.environ:
            self.world_size = int(os.environ['WORLD_SIZE'])

        self.local_rank = 0
        self.world_rank = 0
        if self.world_size > 1:
            dist.init_process_group(backend='nccl',
                                    init_method='env://')
            self.world_rank = dist.get_rank()
            self.local_rank = int(os.environ["LOCAL_RANK"])

        if torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)
            torch.backends.cudnn.benchmark = True
        
        self.log_to_screen = params.log_to_screen and self.world_rank==0
        self.log_to_wandb = params.log_to_wandb and self.world_rank==0
        params['name'] = args.config + '_' + args.run_num
        params['group'] = 'op_' + args.config
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
        else:
            self.device = torch.device('cpu')
        self.params = params
        self.params.device = self.device

    def init_exp_dir(self, exp_dir):
        if self.world_rank==0:
            if not os.path.isdir(exp_dir):
                os.makedirs(exp_dir)
                os.makedirs(os.path.join(exp_dir, 'checkpoints/'))
                os.makedirs(os.path.join(exp_dir, 'wandb/'))
        self.params['experiment_dir'] = os.path.abspath(exp_dir)
        self.params['checkpoint_path'] = os.path.join(exp_dir, 'checkpoints/ckpt.tar')
        #self.params['resuming'] = True if os.path.isfile(self.params.checkpoint_path) else False

    def launch(self):

        if self.sweep_id:
            if self.world_rank==0:
                with wandb.init() as run:
                    hpo_config = wandb.config
                    self.params.update_params(hpo_config)
                    self.modify_bs_for_subsampling()
                    logging.info(self.params.name+'_'+sweep_name_suffix(self.params, self.sweep_id))
                    run.name = self.params.name+'_'+sweep_name_suffix(self.params, self.sweep_id)
                    self.name = run.name
                    self.params.name = self.name
                    exp_dir = os.path.join(*[self.root_dir, 'sweeps', self.sweep_id, self.name, f'bsz{self.params.batch_size}_lr{self.params.lr}_subsample{self.params.subsample}/seed{self.params.seed}'])
                    self.init_exp_dir(exp_dir)
                    logging.info('HPO sweep %s, trial cfg %s'%(self.sweep_id, self.name))
                    self.build_and_run()
            else:
                self.build_and_run()

        else:
            self.modify_bs_for_subsampling()
            exp_dir = os.path.join(*[self.root_dir, f'expts_eps{self.params.max_epochs}', self.config, self.run_num, f'bsz{self.params.batch_size}_lr{self.params.lr}_subsample{self.params.subsample}/seed{self.params.seed}'])
            self.init_exp_dir(exp_dir)
            if self.log_to_wandb:
                wandb.init(dir=os.path.join(exp_dir, "wandb"),
                           config=self.params.params, name=self.params.name, group=self.params.group, project=self.params.project, 
                           entity=self.params.entity, resume=self.params.resuming)
            self.build_and_run()



    def build_and_run(self):

        if self.sweep_id and dist.is_initialized():
            # Broadcast sweep config to other ranks
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            assert self.world_rank == rank
            if rank != 0:
                self.params = None
            self.params = comm.bcast(self.params, root=0)
            self.params.device = self.device # dont broadcast 0s device

        if self.world_rank == 0:
            logging.info(self.params.log())

        set_seed(self.params, self.world_size)

        self.params['global_batch_size'] = self.params.batch_size
        self.params['local_batch_size'] = int(self.params.batch_size//self.world_size)
        self.params['global_valid_batch_size'] = self.params.valid_batch_size
        self.params['local_valid_batch_size'] = int(self.params.valid_batch_size//self.world_size)

        # dump the yaml used
        if self.world_rank == 0:
            hparams = ruamelDict()
            yaml = YAML()
            for key, value in self.params.params.items():
                hparams[str(key)] = str(value)
            with open(os.path.join(self.params['experiment_dir'], 'hyperparams.yaml'), 'w') as hpfile:
                yaml.dump(hparams,  hpfile )

        # data loaders
        self.train_data_loader, self.train_dataset, self.train_sampler = get_data_loader(self.params, self.params.train_path, dist.is_initialized(), train=True, pack=self.params.pack_data)
        self.val_data_loader, self.val_dataset, self.valid_sampler = get_data_loader(self.params, self.params.val_path, dist.is_initialized(), train=False, pack=self.params.pack_data)

        # domain grid
        self.domain = DomainXY(self.params)

        
        if self.params.model == 'fno':
            self.model = models.fno.fno(self.params).to(self.device)
        else:
            assert(False), "Error, model arch invalid."

        if dist.is_initialized():
            self.model = DistributedDataParallel(self.model,
                                                device_ids=[self.local_rank],
                                                output_device=[self.local_rank])



        self.optimizer = set_optimizer(self.params, self.model)

        self.scheduler = set_scheduler(self.params, self.optimizer)

        if self.params.loss_func == "mse":
            self.loss_func = LossMSE(self.params, self.model)
        else:
            assert(False), "Error,  loss func invalid."

        self.iters = 0
        self.startEpoch = 0

        if hasattr(self.params, 'weights'):
            self.params.resuming = False
            logging.info("Loading IC weights %s"%self.params.weights)
            self.load_model(self.params.weights)

        if self.params.resuming:
            logging.info("Loading checkpoint %s"%self.params.checkpoint_path)
            self.restore_checkpoint(self.params.checkpoint_path)

        self.epoch = self.startEpoch
        self.logs = {}
        self.train_loss = self.data_loss = self.bc_loss = self.pde_loss = self.grad = 0.0
        n_params = count_parameters(self.model)
        if self.log_to_screen:
            logging.info(self.model)
            logging.info('number of model parameters: {}'.format(n_params))

        # launch training
        self.train()

    def train(self):
        if self.log_to_screen:
            logging.info("Starting training loop...")
        best_loss = np.inf

        best_epoch = 0
        best_err = 1
        self.logs['best_epoch'] = best_epoch
        plot_figs = self.params.plot_figs

        ### max number of ckpt is 10
        epoch_save_freq = self.params.max_epochs//10
        
        
        for epoch in range(self.startEpoch, self.params.max_epochs):
            self.epoch = epoch
            if dist.is_initialized():
                # shuffles data before every epoch
                self.train_sampler.set_epoch(epoch)
            start = time.time()

            # train
            tr_time = self.train_one_epoch()
            val_time, fields = self.val_one_epoch()
            self.logs['wt_norm'] = self.get_model_wt_norm(self.model)

            if self.params.scheduler == 'reducelr':
                self.scheduler.step(self.logs['train_loss'])
            elif self.params.scheduler == 'cosine':
                self.scheduler.step()

            if self.logs['val_loss'] <= best_loss:
                is_best_loss = True
                best_loss = self.logs['val_loss']
                best_err = self.logs['val_err']
            else:
                is_best_loss = False
            self.logs['best_val_loss'] = best_loss
            self.logs['best_val_err'] = best_err

            best_epoch = self.epoch if is_best_loss else best_epoch
            self.logs['best_epoch'] = best_epoch

            if self.params.save_checkpoint:
                if self.world_rank == 0:
                    #save the best 
                    if is_best_loss:
                        self.save_logs(tag="_best")
                        self.save_checkpoint(self.params.checkpoint_path, is_best=is_best_loss)
                    
                    # save checkpoint every epoch_save_freq epoch
                    if self.epoch % epoch_save_freq == 0:
                        self.save_checkpoint(self.params.checkpoint_path, is_best=False)

                    # save checkpoint every epoch
                    self.save_logs(tag="")
                    
            if self.log_to_wandb:
                # log visualizations every epoch
                if plot_figs:
                    fig = vis_fields(fields, self.params, self.domain)
                    self.logs['vis'] = wandb.Image(fig)
                    plt.close(fig)
                self.logs['learning_rate'] = self.optimizer.param_groups[0]['lr']
                self.logs['time_per_epoch'] = tr_time
                wandb.log(self.logs, step=self.epoch+1)

            if self.log_to_screen:
                logging.info('Time taken for epoch {} is {} sec; with {}/{} in tr/val'.format(self.epoch+1, time.time()-start, tr_time, val_time))
                logging.info('Loss (total = data + bc + pde) {} = {} + {} + {}'.format(self.logs['train_loss'], self.logs['data_loss'],
                self.logs['bc_loss'], self.logs['pde_loss']))

            # check loss for early stopping
            if self.params.early_stopping and self.epoch - best_epoch > self.params.patience:
                break

        if self.log_to_wandb:
            wandb.finish()

    
    def get_model_wt_norm(self, model):
        n = 0
        for p in model.parameters():
            p_norm = p.data.detach().norm(2)
            n += p_norm.item()**2
        n = n**0.5
        return n

    def save_logs(self, tag=""):
        if tag == "_best":
            with open(os.path.join(self.params.experiment_dir, "logs"+tag+".txt"), "w") as f:
                f.write("epoch,{}\n".format(self.epoch))
                for k, v in self.logs.items():
                    f.write("{},{}\n".format(k,v))
        else:
            with open(os.path.join(self.params.experiment_dir, "logs" + tag + ".txt"), "a") as f:
                f.write("\n")
                f.write("epoch,{}\n".format(self.epoch))
                for k, v in self.logs.items():
                    f.write("{},{}\n".format(k, v))     

    def train_one_epoch(self):
        tr_time = 0
        self.model.train()

        # buffers for logs
        logs_buff = torch.zeros((6), dtype=torch.float32, device=self.device)
        self.logs['train_loss'] = logs_buff[0].view(-1)
        self.logs['data_loss'] = logs_buff[1].view(-1)
        self.logs['bc_loss'] = logs_buff[2].view(-1)
        self.logs['pde_loss'] = logs_buff[3].view(-1)
        self.logs['grad'] = logs_buff[4].view(-1)
        self.logs['tr_err'] = logs_buff[5].view(-1)


        for i, (inputs, targets) in enumerate(self.train_data_loader):
            self.iters += 1
            data_start = time.time()
            if not self.params.pack_data: # send to gpu if not already packed in the dataloader
                inputs, targets = inputs.to(self.device), targets.to(self.device)
            tr_start = time.time()

            self.model.zero_grad()
            u = self.model(inputs)

            loss_data = self.loss_func.data(inputs, u, targets)
            loss_pde = self.loss_func.pde(inputs, u, targets)
            loss_bc = self.loss_func.bc(inputs, u, targets)
            loss = loss_data + loss_bc + loss_pde

            loss.backward()
            self.optimizer.step()

            grad_norm = compute_grad_norm(self.model.parameters())
            tr_err = l2_err(u.detach(), targets.detach())
    
            # add all the minibatch losses
            self.logs['train_loss'] += loss.detach()
            self.logs['data_loss'] += loss_data.detach()
            self.logs['bc_loss'] += loss_bc.detach()
            self.logs['pde_loss'] += loss_pde.detach()
            self.logs['grad'] += grad_norm
            self.logs['tr_err'] += tr_err

            tr_time += time.time() - tr_start

        self.logs['train_loss'] /= len(self.train_data_loader)
        self.logs['data_loss'] /= len(self.train_data_loader)
        self.logs['bc_loss'] /= len(self.train_data_loader)
        self.logs['pde_loss'] /= len(self.train_data_loader)
        self.logs['grad'] /= len(self.train_data_loader)
        self.logs['tr_err'] /= len(self.train_data_loader)

        logs_to_reduce = ['train_loss', 'data_loss', 'bc_loss', 'pde_loss', 'grad', 'tr_err']

        if dist.is_initialized():
            for key in logs_to_reduce:
                dist.all_reduce(self.logs[key].detach())
                # todo change loss to unscaled
                self.logs[key] = float(self.logs[key]/dist.get_world_size())

        return tr_time

    def val_one_epoch(self):
        self.model.eval() # need gradients
        #self.model.train() # need gradients
        val_start = time.time()

        logs_buff = torch.zeros((2), dtype=torch.float32, device=self.device)
        self.logs['val_err'] = logs_buff[0].view(-1)
        self.logs['val_loss'] = logs_buff[1].view(-1)
        idx = np.random.randint(0, len(self.val_data_loader))
        img_idx = np.random.randint(0, self.params.local_valid_batch_size)
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(self.val_data_loader):
                if not self.params.pack_data:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                u = self.model(inputs)
                loss_data = self.loss_func.data(inputs, u, targets)
                loss_pde = self.loss_func.pde(inputs, u, targets)
                loss_bc = self.loss_func.bc(inputs, u, targets)
                loss = loss_data + loss_bc + loss_pde
                self.logs['val_err'] += l2_err(u.detach(), targets.detach())
                self.logs['val_loss'] += loss.detach()
                if i == idx: 
                    source = inputs[img_idx,0].detach().cpu().numpy() 
                    soln = targets[img_idx,0].detach().cpu().numpy()
                    pred = u[img_idx,0].detach().cpu().numpy()
                    pde_res = 0*pred
                    temp = 0*pred

        fields = [source, soln, pred, pde_res, temp]

        self.logs['val_loss'] /= len(self.val_data_loader)
        self.logs['val_err'] /= len(self.val_data_loader)
        if dist.is_initialized():
            for key in ['val_loss', 'val_err']:
                dist.all_reduce(self.logs[key].detach())
                self.logs[key] = float(self.logs[key]/dist.get_world_size())

        val_time = time.time() - val_start

        return val_time, fields

    def save_checkpoint(self, checkpoint_path, is_best=False, model=None):
        if not model:
            model = self.model
        if is_best:
            torch.save({'iters': self.iters, 'epoch': self.epoch, 'model_state': model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict(), 'scheduler_state_dict': (self.scheduler.state_dict() if  self.scheduler is not None else None)}, checkpoint_path.replace('.tar', '_best.tar'))
        else:
            torch.save({'iters': self.iters, 'epoch': self.epoch, 'model_state': model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict(), 'scheduler_state_dict': (self.scheduler.state_dict() if  self.scheduler is not None else None)}, checkpoint_path.replace('.tar', f'_{self.epoch}.tar'))

    def restore_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cuda:{}'.format(self.local_rank)) 
        try:
            self.model.load_state_dict(checkpoint['model_state'])
        except:
            new_state_dict = OrderedDict()
            for key, val in checkpoint['model_state'].items():
                name = key[7:]
                new_state_dict[name] = val 
            self.model.load_state_dict(new_state_dict)

        self.iters = checkpoint['iters']
        self.startEpoch = checkpoint['epoch'] + 1
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    def load_model(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cuda:{}'.format(self.local_rank)) 
        try:
            self.model.load_state_dict(checkpoint['model_state'])
        except:
            new_state_dict = OrderedDict()
            for key, val in checkpoint['model_state'].items():
                name = key[7:]
                new_state_dict[name] = val 
            self.model.load_state_dict(new_state_dict)
 
    def switch_off_grad(self, model):
        for param in model.parameters():
            param.requires_grad = False


    def modify_bs_for_subsampling(self):
        '''Reduce batchsize for very small datasets'''
        sz = self.params.subsample
        if sz >= 512:
            fac = np.log2(sz) - 8
            self.params.batch_size = int(128/2**fac)


### write a class to calculate the Hessian matrix / trace Tr(H)

class ModeConnectivityCalculator():
    """Class to calculate Mode Connectivity between neural network models using Bézier curves"""
    
    def __init__(self, params, args=None):
        """
        Initialize ModeConnectivityCalculator
        
        Args:
            params: Configuration parameters (YParams object)
            args: Optional arguments object
        """
        self.params = params
        
        # Set device
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
        else:
            self.device = torch.device('cpu')
        
        self.params.device = self.device
        
        # Set random seed for reproducibility
        set_seed(self.params, world_size=0)  # No distributed training for mode connectivity
        
        # Initialize models - we'll need multiple instances for different checkpoints
        self.model_a = None
        self.model_b = None
        self.model_curve = None  # Model for evaluating points along the curve
        
        # Initialize loss function
        if self.params.loss_func == "mse":
            self.loss_func = LossMSE(self.params, None)  # Will set model later
        else:
            raise ValueError(f"Unsupported loss function: {self.params.loss_func}")
        
        # Set batch size parameters
        self.params['global_batch_size'] = self.params.batch_size
        self.params['local_batch_size'] = self.params.batch_size
        self.params['global_valid_batch_size'] = self.params.valid_batch_size
        self.params['local_valid_batch_size'] = self.params.valid_batch_size
        
        print("ModeConnectivityCalculator initialized")
    
    def _create_model(self):
        """Create a new model instance"""
        if self.params.model == 'fno':
            return models.fno.fno(self.params).to(self.device)
        else:
            raise ValueError(f"Unsupported model type: {self.params.model}")
    
    def load_models(self, checkpoint_a_path, checkpoint_b_path):
        """
        Load two different model checkpoints for connectivity analysis
        
        Args:
            checkpoint_a_path: Path to first model checkpoint
            checkpoint_b_path: Path to second model checkpoint
        """
        print(f"Loading model A from: {checkpoint_a_path}")
        print(f"Loading model B from: {checkpoint_b_path}")
        
        # Create model instances
        self.model_a = self._create_model()
        self.model_b = self._create_model()
        self.model_curve = self._create_model()
        
        # Load checkpoints
        self._load_checkpoint_into_model(self.model_a, checkpoint_a_path)
        self._load_checkpoint_into_model(self.model_b, checkpoint_b_path)
        
        # Set loss function model reference (will be updated for curve evaluation)
        self.loss_func.model = self.model_curve
        
        # Get parameter vectors
        self.params_a = self._get_parameter_vector(self.model_a)
        self.params_b = self._get_parameter_vector(self.model_b)
        self.n_params = len(self.params_a)
        
        print(f"Models loaded successfully with {self.n_params:,} parameters each")
        
    def _load_checkpoint_into_model(self, model, checkpoint_path):
        """Load checkpoint into a specific model"""
        checkpoint = torch.load(checkpoint_path, 
                              map_location=f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu')
        
        try:
            model.load_state_dict(checkpoint['model_state'])
        except:
            # Handle DistributedDataParallel models
            new_state_dict = OrderedDict()
            for key, val in checkpoint['model_state'].items():
                if key.startswith('module.'):
                    name = key[7:]  # Remove 'module.' prefix
                else:
                    name = key
                new_state_dict[name] = val
            model.load_state_dict(new_state_dict)
    
    def _get_parameter_vector(self, model):
        """Extract parameter vector from model"""
        return torch.cat([p.data.reshape(-1) for p in model.parameters() if p.requires_grad])
    
    def _set_parameter_vector(self, model, param_vector):
        """Set model parameters from vector"""
        idx = 0
        for p in model.parameters():
            if p.requires_grad:
                param_size = p.numel()
                p.data.copy_(param_vector[idx:idx+param_size].view(p.shape))
                idx += param_size
    
    def bezier_curve(self, t, control_point=None):
        """
        Compute point on Bézier curve between model A and model B
        Uses quadratic Bézier curve: γ(t) = (1-t)²P₀ + 2(1-t)tP₁ + t²P₂
        
        Args:
            t: Parameter in [0, 1], where 0 = model A, 1 = model B
            control_point: Middle control point (if None, uses midpoint)
        
        Returns:
            torch.Tensor: Parameter vector on the curve
        """
        if control_point is None:
            # Use simple linear interpolation as default
            return (1 - t) * self.params_a + t * self.params_b
        
        # Quadratic Bézier curve
        return ((1 - t)**2 * self.params_a + 
                2 * (1 - t) * t * control_point + 
                t**2 * self.params_b)
    
    def linear_interpolation(self, t):
        """
        Simple linear interpolation between model A and model B
        
        Args:
            t: Parameter in [0, 1]
        
        Returns:
            torch.Tensor: Parameter vector on the line
        """
        return (1 - t) * self.params_a + t * self.params_b
    
    def evaluate_loss_at_point(self, param_vector, data_loader, max_batches=None):
        """
        Evaluate loss at a specific point in parameter space
        
        Args:
            param_vector: Parameter vector to evaluate
            data_loader: Data loader for loss computation
            max_batches: Maximum number of batches to process
        
        Returns:
            float: Average loss value
        """
        # Set model parameters
        self._set_parameter_vector(self.model_curve, param_vector)
        self.model_curve.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(data_loader):
                if max_batches is not None and batch_idx >= max_batches:
                    break
                
                if not hasattr(self.params, 'pack_data') or not self.params.pack_data:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Compute forward pass
                u = self.model_curve(inputs)
                
                # Compute loss components
                loss_data = self.loss_func.data(inputs, u, targets)
                loss_pde = self.loss_func.pde(inputs, u, targets)
                loss_bc = self.loss_func.bc(inputs, u, targets)
                loss = loss_data + loss_bc + loss_pde
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / max(num_batches, 1)
    
    def compute_loss_along_curve(self, data_loader, num_points=21, control_point=None, max_batches=None):
        """
        Compute loss at multiple points along the Bézier curve
        
        Args:
            data_loader: Data loader for loss computation
            num_points: Number of points to evaluate along the curve
            control_point: Control point for Bézier curve (None for linear)
            max_batches: Maximum number of batches per evaluation
        
        Returns:
            tuple: (t_values, loss_values)
        """
        t_values = torch.linspace(0, 1, num_points)
        loss_values = []
        
        print(f"Evaluating loss at {num_points} points along the curve...")
        
        for i, t in enumerate(t_values):
            # Get parameter vector at point t
            if control_point is not None:
                param_vector = self.bezier_curve(t.item(), control_point)
            else:
                param_vector = self.linear_interpolation(t.item())
            
            # Evaluate loss
            loss = self.evaluate_loss_at_point(param_vector, data_loader, max_batches)
            loss_values.append(loss)
            
            if i % max(1, num_points // 10) == 0:
                print(f"  Point {i+1}/{num_points}: t={t:.2f}, loss={loss:.6f}")
        
        return t_values.numpy(), np.array(loss_values)
    
    def find_optimal_t(self, data_loader, target_loss, num_points=21, control_point=None, max_batches=None):
        """
        Find t* that minimizes |target_loss - L(γ(t))|
        
        Args:
            data_loader: Data loader for loss computation
            target_loss: Target loss value (typically average of endpoint losses)
            num_points: Number of points to search
            control_point: Control point for Bézier curve
            max_batches: Maximum batches per evaluation
        
        Returns:
            tuple: (optimal_t, optimal_loss, min_difference)
        """
        t_values, loss_values = self.compute_loss_along_curve(
            data_loader, num_points, control_point, max_batches)
        
        # Find t that minimizes |target_loss - loss|
        differences = np.abs(loss_values - target_loss)
        min_idx = np.argmin(differences)
        
        optimal_t = t_values[min_idx]
        optimal_loss = loss_values[min_idx]
        min_difference = differences[min_idx]
        
        return optimal_t, optimal_loss, min_difference
    
    def compute_mode_connectivity(self, data_loader, curve_type='linear', optimize_control_point=False, 
                                 num_curve_points=21, max_batches=None):
        """
        Compute mode connectivity between the two loaded models
        
        Args:
            data_loader: Data loader for loss computation
            curve_type: 'linear' or 'bezier'
            optimize_control_point: Whether to optimize the Bézier control point
            num_curve_points: Number of points to evaluate along curve
            max_batches: Maximum batches per evaluation
        
        Returns:
            dict: Results containing mode connectivity and analysis
        """
        if self.model_a is None or self.model_b is None:
            raise ValueError("Must load two models first using load_models()")
        
        print("Computing mode connectivity...")
        print(f"Curve type: {curve_type}")
        print(f"Optimize control point: {optimize_control_point}")
        
        # Compute endpoint losses
        print("Evaluating endpoint losses...")
        loss_a = self.evaluate_loss_at_point(self.params_a, data_loader, max_batches)
        loss_b = self.evaluate_loss_at_point(self.params_b, data_loader, max_batches)
        avg_endpoint_loss = 0.5 * (loss_a + loss_b)
        
        print(f"Loss A: {loss_a:.6f}")
        print(f"Loss B: {loss_b:.6f}")
        print(f"Average endpoint loss: {avg_endpoint_loss:.6f}")
        
        # Initialize control point
        control_point = None
        if curve_type == 'bezier':
            if optimize_control_point:
                # Start with midpoint and optimize
                control_point = 0.5 * (self.params_a + self.params_b)
                control_point = self._optimize_control_point(control_point, data_loader, 
                                                           avg_endpoint_loss, max_batches)
            else:
                # Use simple midpoint
                control_point = 0.5 * (self.params_a + self.params_b)
        
        # Find optimal t*
        print("Finding optimal t*...")
        optimal_t, optimal_loss, min_diff = self.find_optimal_t(
            data_loader, avg_endpoint_loss, num_curve_points, control_point, max_batches)
        
        # Compute mode connectivity
        mode_connectivity = avg_endpoint_loss - optimal_loss
        
        # Get full curve for analysis
        t_values, loss_values = self.compute_loss_along_curve(
            data_loader, num_curve_points, control_point, max_batches)
        
        results = {
            'mode_connectivity': mode_connectivity,
            'loss_a': loss_a,
            'loss_b': loss_b,
            'avg_endpoint_loss': avg_endpoint_loss,
            'optimal_t': optimal_t,
            'optimal_loss': optimal_loss,
            'min_difference': min_diff,
            'curve_type': curve_type,
            'optimize_control_point': optimize_control_point,
            't_values': t_values,
            'loss_values': loss_values,
            'loss_barrier_height': np.max(loss_values) - avg_endpoint_loss,
            'loss_curve_min': np.min(loss_values),
            'loss_curve_max': np.max(loss_values),
            'parameter_distance': torch.norm(self.params_b - self.params_a).item(),
        }
        
        # Interpret results
        self._interpret_results(results)
        
        return results
    
    def _optimize_control_point(self, initial_control_point, data_loader, target_loss, 
                               max_batches=None, num_iterations=50, lr=0.01):
        """
        Optimize the Bézier curve control point to minimize loss barrier
        
        Args:
            initial_control_point: Starting control point
            data_loader: Data loader
            target_loss: Target loss value
            max_batches: Maximum batches per evaluation
            num_iterations: Number of optimization iterations
            lr: Learning rate
        
        Returns:
            torch.Tensor: Optimized control point
        """
        print(f"Optimizing control point for {num_iterations} iterations...")
        
        control_point = initial_control_point.clone().requires_grad_(True)
        optimizer = torch.optim.Adam([control_point], lr=lr)
        
        best_loss = float('inf')
        best_control_point = control_point.clone()
        
        for iteration in range(num_iterations):
            optimizer.zero_grad()
            
            # Evaluate loss at midpoint of curve (t=0.5)
            param_vector = self.bezier_curve(0.5, control_point)
            loss = self.evaluate_loss_at_point(param_vector, data_loader, max_batches)
            
            # Minimize the loss at the midpoint
            loss_tensor = torch.tensor(loss, requires_grad=True)
            loss_tensor.backward()
            
            # Manual gradient for control point (approximate)
            # This is a simplified approach - in practice, you'd want to compute exact gradients
            if control_point.grad is not None:
                control_point.grad.zero_()
            
            # Simple gradient approximation
            eps = 1e-4
            control_point_plus = control_point + eps
            control_point_minus = control_point - eps
            
            param_plus = self.bezier_curve(0.5, control_point_plus)
            param_minus = self.bezier_curve(0.5, control_point_minus)
            
            loss_plus = self.evaluate_loss_at_point(param_plus, data_loader, max_batches)
            loss_minus = self.evaluate_loss_at_point(param_minus, data_loader, max_batches)
            
            approx_grad = (loss_plus - loss_minus) / (2 * eps)
            
            # Update control point
            with torch.no_grad():
                control_point -= lr * approx_grad * torch.ones_like(control_point)
            
            if loss < best_loss:
                best_loss = loss
                best_control_point = control_point.clone()
            
            if iteration % 10 == 0:
                print(f"  Iteration {iteration}: loss = {loss:.6f}")
        
        print(f"Control point optimization completed. Best loss: {best_loss:.6f}")
        return best_control_point.detach()
    
    def _interpret_results(self, results):
        """Print interpretation of mode connectivity results"""
        mc = results['mode_connectivity']
        
        print("\n" + "="*60)
        print("MODE CONNECTIVITY ANALYSIS")
        print("="*60)
        print(f"Mode Connectivity (mc): {mc:.6f}")
        print(f"Loss A: {results['loss_a']:.6f}")
        print(f"Loss B: {results['loss_b']:.6f}")
        print(f"Average endpoint loss: {results['avg_endpoint_loss']:.6f}")
        print(f"Optimal t*: {results['optimal_t']:.3f}")
        print(f"Loss at t*: {results['optimal_loss']:.6f}")
        print(f"Loss barrier height: {results['loss_barrier_height']:.6f}")
        print(f"Parameter distance: {results['parameter_distance']:.6f}")
        
        print("\nInterpretation:")
        if mc < -0.001:  # Small threshold for numerical precision
            print("❌ mc < 0: POOR CONNECTIVITY")
            print("   Loss barrier exists between models")
            print("   Models are in disconnected regions of loss landscape")
        elif mc > 0.001:
            print("⚠️  mc > 0: SUSPICIOUS CONNECTIVITY")
            print("   Lower loss regions exist between models")
            print("   May indicate poor training or overfitting")
        else:
            print("✅ mc ≈ 0: GOOD CONNECTIVITY")
            print("   Models are well-connected in parameter space")
            print("   Indicates smooth loss landscape")
        
        print("="*60)
    
    def analyze_connectivity_properties(self, data_loader, max_batches=None):
        """
        Comprehensive analysis of connectivity properties
        
        Args:
            data_loader: Data loader for analysis
            max_batches: Maximum batches per evaluation
        
        Returns:
            dict: Comprehensive analysis results
        """
        if self.model_a is None or self.model_b is None:
            raise ValueError("Must load two models first using load_models()")
        
        print("Performing comprehensive connectivity analysis...")
        
        results = {}
        
        # Analyze linear interpolation
        print("\n1. Linear interpolation analysis...")
        linear_results = self.compute_mode_connectivity(
            data_loader, curve_type='linear', max_batches=max_batches)
        results['linear'] = linear_results
        
        # Analyze Bézier curve with midpoint
        print("\n2. Bézier curve (midpoint control) analysis...")
        bezier_results = self.compute_mode_connectivity(
            data_loader, curve_type='bezier', optimize_control_point=False, max_batches=max_batches)
        results['bezier_midpoint'] = bezier_results
        
        # Analyze optimized Bézier curve (if computational budget allows)
        if max_batches is None or max_batches >= 5:
            print("\n3. Optimized Bézier curve analysis...")
            try:
                optimized_results = self.compute_mode_connectivity(
                    data_loader, curve_type='bezier', optimize_control_point=True, max_batches=max_batches)
                results['bezier_optimized'] = optimized_results
            except Exception as e:
                print(f"Warning: Could not compute optimized Bézier curve: {e}")
        
        # Compute summary statistics
        print("\n" + "="*60)
        print("CONNECTIVITY COMPARISON")
        print("="*60)
        
        for method, res in results.items():
            print(f"{method.upper()}:")
            print(f"  Mode connectivity: {res['mode_connectivity']:.6f}")
            print(f"  Loss barrier height: {res['loss_barrier_height']:.6f}")
            print(f"  Optimal t*: {res['optimal_t']:.3f}")
            print()
        
        return results
    
    def plot_loss_landscape(self, results, save_path=None):
        """
        Plot the loss landscape along the connectivity curve
        
        Args:
            results: Results from compute_mode_connectivity
            save_path: Optional path to save the plot
        
        Returns:
            matplotlib.figure.Figure: The plot figure
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Matplotlib not available for plotting")
            return None
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Plot loss curve
        t_vals = results['t_values']
        loss_vals = results['loss_values']
        
        ax.plot(t_vals, loss_vals, 'b-', linewidth=2, label='Loss curve')
        
        # Mark endpoints
        ax.plot(0, results['loss_a'], 'ro', markersize=8, label=f'Model A (loss={results["loss_a"]:.4f})')
        ax.plot(1, results['loss_b'], 'go', markersize=8, label=f'Model B (loss={results["loss_b"]:.4f})')
        
        # Mark optimal point
        ax.plot(results['optimal_t'], results['optimal_loss'], 'ko', markersize=8, 
                label=f't*={results["optimal_t"]:.3f} (loss={results["optimal_loss"]:.4f})')
        
        # Mark average endpoint loss
        ax.axhline(y=results['avg_endpoint_loss'], color='orange', linestyle='--', alpha=0.7,
                  label=f'Average endpoint loss ({results["avg_endpoint_loss"]:.4f})')
        
        # Formatting
        ax.set_xlabel('t (interpolation parameter)')
        ax.set_ylabel('Loss')
        ax.set_title(f'Mode Connectivity Analysis\nmc = {results["mode_connectivity"]:.6f} ({results["curve_type"]})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add interpretation text
        mc = results['mode_connectivity']
        if mc < -0.001:
            interpretation = "Poor connectivity (loss barrier)"
            color = 'red'
        elif mc > 0.001:
            interpretation = "Suspicious connectivity (lower loss path)"
            color = 'orange'
        else:
            interpretation = "Good connectivity"
            color = 'green'
        
        ax.text(0.02, 0.98, f'Interpretation: {interpretation}', 
                transform=ax.transAxes, fontsize=10, 
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.2),
                verticalalignment='top')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        return fig
    
    @classmethod
    def from_config(cls, params):
        """
        Create ModeConnectivityCalculator from config parameters
        
        Args:
            params: Configuration parameters (YParams object)
            
        Returns:
            ModeConnectivityCalculator: Initialized calculator
        """
        return cls(params)
    
    def compute_pairwise_connectivity(self, checkpoint_paths, data_loader, max_batches=None):
        """
        Compute mode connectivity between all pairs of checkpoints
        
        Args:
            checkpoint_paths: List of checkpoint paths
            data_loader: Data loader for loss computation
            max_batches: Maximum batches per evaluation
        
        Returns:
            dict: Pairwise connectivity results
        """
        n_models = len(checkpoint_paths)
        connectivity_matrix = np.zeros((n_models, n_models))
        results = {}
        
        print(f"Computing pairwise connectivity for {n_models} models...")
        
        for i in range(n_models):
            for j in range(i+1, n_models):
                print(f"\nAnalyzing pair {i+1}-{j+1}: {checkpoint_paths[i]} <-> {checkpoint_paths[j]}")
                
                # Load models
                self.load_models(checkpoint_paths[i], checkpoint_paths[j])
                
                # Compute connectivity
                result = self.compute_mode_connectivity(data_loader, max_batches=max_batches)
                
                # Store results
                connectivity_matrix[i, j] = result['mode_connectivity']
                connectivity_matrix[j, i] = result['mode_connectivity']  # Symmetric
                results[f'{i}_{j}'] = result
        
        return {
            'connectivity_matrix': connectivity_matrix,
            'pairwise_results': results,
            'checkpoint_paths': checkpoint_paths
        }

class CKACalculator():
    """Class to calculate Centered Kernel Alignment (CKA) similarity between neural network representations"""
    
    def __init__(self, params, args=None):
        """
        Initialize CKACalculator
        
        Args:
            params: Configuration parameters (YParams object)
            args: Optional arguments object
        """
        self.params = params
        
        # Set device
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
        else:
            self.device = torch.device('cpu')
        
        self.params.device = self.device
        
        # Set random seed for reproducibility
        set_seed(self.params, world_size=0)  # No distributed training for CKA
        
        # Initialize models - we'll need multiple instances for different checkpoints
        self.model_a = None
        self.model_b = None
        
        # Initialize loss function (for validation purposes)
        if self.params.loss_func == "mse":
            self.loss_func = LossMSE(self.params, None)  # Will set model later
        else:
            raise ValueError(f"Unsupported loss function: {self.params.loss_func}")
        
        # Set batch size parameters
        self.params['global_batch_size'] = self.params.batch_size
        self.params['local_batch_size'] = self.params.batch_size
        self.params['global_valid_batch_size'] = self.params.valid_batch_size
        self.params['local_valid_batch_size'] = self.params.valid_batch_size
        
        print("CKACalculator initialized")
    
    def _create_model(self):
        """Create a new model instance"""
        if self.params.model == 'fno':
            return models.fno.fno(self.params).to(self.device)
        else:
            raise ValueError(f"Unsupported model type: {self.params.model}")
    
    def load_models(self, checkpoint_a_path, checkpoint_b_path):
        """
        Load two different model checkpoints for CKA analysis
        
        Args:
            checkpoint_a_path: Path to first model checkpoint
            checkpoint_b_path: Path to second model checkpoint
        """
        print(f"Loading model A from: {checkpoint_a_path}")
        print(f"Loading model B from: {checkpoint_b_path}")
        
        # Create model instances
        self.model_a = self._create_model()
        self.model_b = self._create_model()
        
        # Load checkpoints
        self._load_checkpoint_into_model(self.model_a, checkpoint_a_path)
        self._load_checkpoint_into_model(self.model_b, checkpoint_b_path)
        
        # Set models to evaluation mode
        self.model_a.eval()
        self.model_b.eval()
        
        print("Models loaded successfully for CKA analysis")
        
    def _load_checkpoint_into_model(self, model, checkpoint_path):
        """Load checkpoint into a specific model"""
        checkpoint = torch.load(checkpoint_path, 
                              map_location=f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu')
        
        try:
            model.load_state_dict(checkpoint['model_state'])
        except:
            # Handle DistributedDataParallel models
            new_state_dict = OrderedDict()
            for key, val in checkpoint['model_state'].items():
                if key.startswith('module.'):
                    name = key[7:]  # Remove 'module.' prefix
                else:
                    name = key
                new_state_dict[name] = val
            model.load_state_dict(new_state_dict)
    
    def extract_representations(self, model, data_loader, max_batches=None, layer_name=None):
        """
        Extract representations (outputs/features) from a model
        
        Args:
            model: The neural network model
            data_loader: Data loader for representation extraction
            max_batches: Maximum number of batches to process
            layer_name: Specific layer to extract features from (None for final output)
        
        Returns:
            torch.Tensor: Concatenated representations [N, D] where N is number of samples
        """
        model.eval()
        representations = []
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(data_loader):
                if max_batches is not None and batch_idx >= max_batches:
                    break
                
                if not hasattr(self.params, 'pack_data') or not self.params.pack_data:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Forward pass to get representations
                if layer_name is None:
                    # Extract final output
                    outputs = model(inputs)
                    # Flatten to [batch_size, -1]
                    batch_repr = outputs.view(outputs.size(0), -1)
                else:
                    # Extract intermediate layer features (would need hooks implementation)
                    raise NotImplementedError("Intermediate layer extraction not implemented yet")
                
                representations.append(batch_repr.cpu())
        
        # Concatenate all representations
        all_representations = torch.cat(representations, dim=0)
        return all_representations
    
    def gram_linear(self, X):
        """
        Compute linear Gram matrix: K = X @ X.T
        
        Args:
            X: Input matrix [N, D]
        
        Returns:
            torch.Tensor: Gram matrix [N, N]
        """
        return torch.mm(X, X.t())
    
    def gram_rbf(self, X, sigma=1.0):
        """
        Compute RBF Gram matrix: K_ij = exp(-||x_i - x_j||^2 / (2*sigma^2))
        
        Args:
            X: Input matrix [N, D]
            sigma: RBF bandwidth
        
        Returns:
            torch.Tensor: RBF Gram matrix [N, N]
        """
        # Compute pairwise squared distances
        X_norm = (X**2).sum(dim=1, keepdim=True)
        distances = X_norm + X_norm.t() - 2 * torch.mm(X, X.t())
        
        # Compute RBF kernel
        return torch.exp(-distances / (2 * sigma**2))
    
    def center_gram_matrix(self, K):
        """
        Center the Gram matrix: K_c = HKH where H = I - (1/n)11^T
        
        Args:
            K: Gram matrix [N, N]
        
        Returns:
            torch.Tensor: Centered Gram matrix [N, N]
        """
        n = K.size(0)
        H = torch.eye(n, device=K.device) - torch.ones(n, n, device=K.device) / n
        return torch.mm(torch.mm(H, K), H)
    
    def hsic(self, K, L):
        """
        Compute Hilbert-Schmidt Independence Criterion (HSIC)
        HSIC(K, L) = (1/(n-1)^2) * tr(KHLH)
        
        Args:
            K: First Gram matrix [N, N]
            L: Second Gram matrix [N, N]
        
        Returns:
            float: HSIC value
        """
        n = K.size(0)
        
        # Center the Gram matrices
        K_c = self.center_gram_matrix(K)
        L_c = self.center_gram_matrix(L)
        
        # Compute HSIC
        hsic_value = torch.trace(torch.mm(K_c, L_c)) / ((n - 1) ** 2)
        return hsic_value.item()
    
    def cka_similarity(self, X, Y, kernel='linear', sigma=1.0):
        """
        Compute CKA similarity between two representation matrices
        
        CKA = HSIC(K, L) / sqrt(HSIC(K, K) * HSIC(L, L))
        
        Args:
            X: First representation matrix [N, D1]
            Y: Second representation matrix [N, D2]
            kernel: Kernel type ('linear' or 'rbf')
            sigma: RBF bandwidth (only used for RBF kernel)
        
        Returns:
            float: CKA similarity score
        """
        # Ensure tensors are on the same device
        if X.device != Y.device:
            Y = Y.to(X.device)
        
        # Compute Gram matrices
        if kernel == 'linear':
            K = self.gram_linear(X)
            L = self.gram_linear(Y)
        elif kernel == 'rbf':
            K = self.gram_rbf(X, sigma)
            L = self.gram_rbf(Y, sigma)
        else:
            raise ValueError(f"Unsupported kernel type: {kernel}")
        
        # Compute HSIC values
        hsic_kl = self.hsic(K, L)
        hsic_kk = self.hsic(K, K)
        hsic_ll = self.hsic(L, L)
        
        # Avoid division by zero
        denominator = np.sqrt(hsic_kk * hsic_ll)
        if denominator < 1e-12:
            return 0.0
        
        cka_score = hsic_kl / denominator
        return cka_score
    
    def compute_cka_between_models(self, data_loader, kernel='linear', sigma=1.0, 
                                  max_batches=None, layer_name=None):
        """
        Compute CKA similarity between two loaded models
        
        Args:
            data_loader: Data loader for representation extraction
            kernel: Kernel type ('linear' or 'rbf')
            sigma: RBF bandwidth
            max_batches: Maximum number of batches to process
            layer_name: Layer to extract features from (None for final output)
        
        Returns:
            dict: Results containing CKA score and analysis
        """
        if self.model_a is None or self.model_b is None:
            raise ValueError("Must load two models first using load_models()")
        
        print(f"Computing CKA similarity using {kernel} kernel...")
        
        # Extract representations from both models
        print(f"Extracting representations from Model A...")
        repr_a = self.extract_representations(self.model_a, data_loader, max_batches, layer_name)
        
        print(f"Extracting representations from Model B...")
        repr_b = self.extract_representations(self.model_b, data_loader, max_batches, layer_name)
        
        # Ensure same number of samples
        min_samples = min(repr_a.size(0), repr_b.size(0))
        repr_a = repr_a[:min_samples].to(self.device)
        repr_b = repr_b[:min_samples].to(self.device)
        
        print(f"Computing CKA on {min_samples} samples...")
        print(f"Representation A shape: {repr_a.shape}")
        print(f"Representation B shape: {repr_b.shape}")
        
        # Compute CKA similarity
        cka_score = self.cka_similarity(repr_a, repr_b, kernel=kernel, sigma=sigma)
        
        # Additional statistics
        repr_a_norm = torch.norm(repr_a, dim=1).mean().item()
        repr_b_norm = torch.norm(repr_b, dim=1).mean().item()
        cosine_sim = torch.nn.functional.cosine_similarity(
            repr_a.mean(dim=0, keepdim=True), 
            repr_b.mean(dim=0, keepdim=True)
        ).item()
        
        results = {
            'cka_score': cka_score,
            'kernel_type': kernel,
            'sigma': sigma if kernel == 'rbf' else None,
            'num_samples': min_samples,
            'repr_a_shape': list(repr_a.shape),
            'repr_b_shape': list(repr_b.shape),
            'repr_a_mean_norm': repr_a_norm,
            'repr_b_mean_norm': repr_b_norm,
            'mean_cosine_similarity': cosine_sim,
            'layer_name': layer_name if layer_name else 'output'
        }
        
        print(f"CKA similarity: {cka_score:.6f}")
        
        return results
    
    def analyze_cka_properties(self, data_loader, max_batches=None):
        """
        Comprehensive CKA analysis with different kernels and properties
        
        Args:
            data_loader: Data loader for analysis
            max_batches: Maximum number of batches to process
        
        Returns:
            dict: Comprehensive CKA analysis results
        """
        if self.model_a is None or self.model_b is None:
            raise ValueError("Must load two models first using load_models()")
        
        print("Performing comprehensive CKA analysis...")
        
        results = {}
        
        # Analyze with linear kernel
        print("\n1. Linear kernel CKA...")
        linear_results = self.compute_cka_between_models(
            data_loader, kernel='linear', max_batches=max_batches)
        results['linear'] = linear_results
        
        # Analyze with RBF kernel (multiple sigma values)
        print("\n2. RBF kernel CKA...")
        rbf_sigmas = [0.1, 0.5, 1.0, 2.0, 5.0]
        rbf_results = {}
        
        for sigma in rbf_sigmas:
            print(f"   RBF kernel with sigma={sigma}...")
            rbf_result = self.compute_cka_between_models(
                data_loader, kernel='rbf', sigma=sigma, max_batches=max_batches)
            rbf_results[f'sigma_{sigma}'] = rbf_result
        
        results['rbf'] = rbf_results
        
        # Summary statistics
        print("\n" + "="*60)
        print("CKA ANALYSIS SUMMARY")
        print("="*60)
        print(f"Linear CKA: {results['linear']['cka_score']:.6f}")
        
        print("RBF CKA scores:")
        for sigma_key, rbf_res in rbf_results.items():
            sigma_val = sigma_key.replace('sigma_', '')
            print(f"  σ={sigma_val}: {rbf_res['cka_score']:.6f}")
        
        # Interpretation
        linear_cka = results['linear']['cka_score']
        print("\nInterpretation:")
        if linear_cka > 0.9:
            print("🔍 Very High CKA (>0.9): Representations are nearly identical")
        elif linear_cka > 0.7:
            print("✅ High CKA (0.7-0.9): Representations are very similar")
        elif linear_cka > 0.5:
            print("⚠️  Moderate CKA (0.5-0.7): Representations show some similarity")
        elif linear_cka > 0.3:
            print("❌ Low CKA (0.3-0.5): Representations are quite different")
        else:
            print("💥 Very Low CKA (<0.3): Representations are very different")
        
        print("="*60)
        
        return results
    
    def compute_layerwise_cka(self, data_loader, layer_names=None, max_batches=None):
        """
        Compute CKA similarity at different layers of the networks
        
        Args:
            data_loader: Data loader for analysis
            layer_names: List of layer names to analyze (None for just output)
            max_batches: Maximum number of batches to process
        
        Returns:
            dict: Layer-wise CKA results
        """
        # Note: This is a placeholder implementation
        # Full implementation would require hooking into intermediate layers
        
        print("Layer-wise CKA analysis (output layer only for now)...")
        
        if layer_names is None:
            layer_names = ['output']
        
        results = {}
        
        for layer_name in layer_names:
            print(f"Analyzing layer: {layer_name}")
            if layer_name == 'output':
                layer_results = self.compute_cka_between_models(
                    data_loader, kernel='linear', max_batches=max_batches, layer_name=None)
            else:
                # Placeholder for intermediate layers
                print(f"Warning: Intermediate layer '{layer_name}' analysis not implemented")
                continue
            
            results[layer_name] = layer_results
        
        return results
    
    def compute_pairwise_cka(self, checkpoint_paths, data_loader, kernel='linear', 
                           sigma=1.0, max_batches=None):
        """
        Compute CKA similarity between all pairs of checkpoints
        
        Args:
            checkpoint_paths: List of checkpoint paths
            data_loader: Data loader for representation extraction
            kernel: Kernel type
            sigma: RBF bandwidth
            max_batches: Maximum number of batches to process
        
        Returns:
            dict: Pairwise CKA results
        """
        n_models = len(checkpoint_paths)
        cka_matrix = np.zeros((n_models, n_models))
        results = {}
        
        print(f"Computing pairwise CKA for {n_models} models...")
        
        for i in range(n_models):
            for j in range(i+1, n_models):
                print(f"\nAnalyzing pair {i+1}-{j+1}: {checkpoint_paths[i]} <-> {checkpoint_paths[j]}")
                
                # Load models
                self.load_models(checkpoint_paths[i], checkpoint_paths[j])
                
                # Compute CKA
                result = self.compute_cka_between_models(
                    data_loader, kernel=kernel, sigma=sigma, max_batches=max_batches)
                
                # Store results
                cka_matrix[i, j] = result['cka_score']
                cka_matrix[j, i] = result['cka_score']  # Symmetric
                results[f'{i}_{j}'] = result
        
        # Diagonal is 1.0 (self-similarity)
        np.fill_diagonal(cka_matrix, 1.0)
        
        return {
            'cka_matrix': cka_matrix,
            'pairwise_results': results,
            'checkpoint_paths': checkpoint_paths
        }
    
    def plot_cka_heatmap(self, cka_matrix, checkpoint_names=None, save_path=None):
        """
        Plot CKA similarity heatmap
        
        Args:
            cka_matrix: CKA similarity matrix
            checkpoint_names: Names for checkpoints (optional)
            save_path: Path to save the plot
        
        Returns:
            matplotlib.figure.Figure: The plot figure
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            print("Matplotlib/Seaborn not available for plotting")
            return None
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # Create heatmap
        im = ax.imshow(cka_matrix, cmap='viridis', vmin=0, vmax=1)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('CKA Similarity', rotation=270, labelpad=20)
        
        # Set labels
        if checkpoint_names:
            ax.set_xticks(range(len(checkpoint_names)))
            ax.set_yticks(range(len(checkpoint_names)))
            ax.set_xticklabels(checkpoint_names, rotation=45, ha='right')
            ax.set_yticklabels(checkpoint_names)
        
        # Add text annotations
        for i in range(cka_matrix.shape[0]):
            for j in range(cka_matrix.shape[1]):
                text = ax.text(j, i, f'{cka_matrix[i, j]:.3f}',
                             ha="center", va="center", color="white" if cka_matrix[i, j] < 0.5 else "black")
        
        ax.set_title('CKA Similarity Matrix')
        ax.set_xlabel('Model')
        ax.set_ylabel('Model')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"CKA heatmap saved to {save_path}")
        
        return fig
    
    @classmethod
    def from_config(cls, params):
        """
        Create CKACalculator from config parameters
        
        Args:
            params: Configuration parameters (YParams object)
            
        Returns:
            CKACalculator: Initialized calculator
        """
        return cls(params)
    
    def load_model(self, checkpoint_path):
        """
        Load single model weights from checkpoint (for model_a)
        
        Args:
            checkpoint_path: Path to the checkpoint file
        """
        print(f"Loading model weights from {checkpoint_path}")
        
        if self.model_a is None:
            self.model_a = self._create_model()
        
        self._load_checkpoint_into_model(self.model_a, checkpoint_path)
        print("Model weights loaded successfully")
    
    def restore_checkpoint(self, checkpoint_path):
        """
        Restore checkpoint and return metadata
        
        Args:
            checkpoint_path: Path to the checkpoint file
        
        Returns:
            dict: Checkpoint metadata
        """
        print(f"Restoring checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, 
                              map_location=f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu')
        
        if self.model_a is None:
            self.model_a = self._create_model()
        
        self._load_checkpoint_into_model(self.model_a, checkpoint_path)
        
        # Return checkpoint metadata
        metadata = {
            'epoch': checkpoint.get('epoch', 0),
            'iters': checkpoint.get('iters', 0)
        }
        
        print(f"Checkpoint restored successfully (epoch: {metadata['epoch']}, iters: {metadata['iters']})")
        return metadata

def orthonormalization(v, v_list):
    """
    Orthonormalize vector v against a list of vectors v_list
    Args:
        v: Vector to orthonormalize (list of tensors)
        v_list: List of vectors to orthonormalize against
    Returns:
        Orthonormalized vector
    """
    for v_ref in v_list:
        # Compute dot product
        dot = sum(torch.sum(a * b) for a, b in zip(v, v_ref))
        # Subtract projection
        v = [a - dot * b for a, b in zip(v, v_ref)]

    # Normalize
    norm = sum(torch.sum(a * a) for a in v) ** 0.5
    v = [a / (norm + 1e-12) for a in v]

    return v


class LossLandscapeCalculator():
    """Class to calculate and visualize loss landscapes using Hessian eigenvectors"""

    def __init__(self, params, args=None):
        """
        Initialize LossLandscapeCalculator

        Args:
            params: Configuration parameters (YParams object)
            args: Optional arguments object
        """
        self.params = params

        # Set device
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
        else:
            self.device = torch.device('cpu')

        self.params.device = self.device

        # Set random seed for reproducibility
        set_seed(self.params, world_size=0)

        # Initialize model
        if self.params.model == 'fno':
            self.model = models.fno.fno(self.params).to(self.device)
        else:
            raise ValueError(f"Unsupported model type: {self.params.model}")

        # Set batch size parameters
        self.params['global_batch_size'] = self.params.batch_size
        self.params['local_batch_size'] = self.params.batch_size
        self.params['global_valid_batch_size'] = self.params.valid_batch_size
        self.params['local_valid_batch_size'] = self.params.valid_batch_size

        # Initialize loss function
        if self.params.loss_func == "mse":
            self.loss_func = LossMSE(self.params, self.model)
        else:
            raise ValueError(f"Unsupported loss function: {self.params.loss_func}")

        print("LossLandscapeCalculator initialized")

    def _create_model(self):
        """Create a new model instance"""
        if self.params.model == 'fno':
            return models.fno.fno(self.params).to(self.device)
        else:
            raise ValueError(f"Unsupported model type: {self.params.model}")

    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path,
                              map_location=f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu')

        try:
            self.model.load_state_dict(checkpoint['model_state'])
        except:
            # Handle DistributedDataParallel models
            new_state_dict = OrderedDict()
            for key, val in checkpoint['model_state'].items():
                if key.startswith('module.'):
                    name = key[7:]  # Remove 'module.' prefix
                else:
                    name = key
                new_state_dict[name] = val
            self.model.load_state_dict(new_state_dict)

        print("Checkpoint loaded successfully")

    def compute_directions_from_hessian(self, data_loader, max_batches=1):
        """
        Compute top-2 Hessian eigenvectors as directions for landscape exploration

        Args:
            data_loader: DataLoader for Hessian computation
            max_batches: Number of batches to use for Hessian

        Returns:
            tuple: (d1, d2) - two orthonormal direction vectors
        """
        from utils.hessian import hessian

        print("Computing Hessian eigenvectors for directions...")

        # Prepare single batch for Hessian
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            if not hasattr(self.params, 'pack_data') or not self.params.pack_data:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
            break

        # Create criterion for Hessian
        def criterion(outputs, targets_inner):
            dummy_inputs = torch.zeros_like(targets_inner)
            loss_data = self.loss_func.data(dummy_inputs, outputs, targets_inner)
            loss_pde = self.loss_func.pde(dummy_inputs, outputs, targets_inner)
            loss_bc = self.loss_func.bc(dummy_inputs, outputs, targets_inner)
            return loss_data + loss_pde + loss_bc

        # Compute Hessian
        hessian_calc = hessian(
            model=self.model,
            criterion=criterion,
            data=(inputs, targets),
            dataloader=None,
            cuda=torch.cuda.is_available()
        )

        # Get top-2 eigenvectors
        eigenvalues, eigenvectors = hessian_calc.eigenvalues(maxIter=100, tol=1e-3, top_n=2)

        print(f"Top-2 eigenvalues: {eigenvalues}")

        # Orthonormalize second eigenvector against first
        d1 = eigenvectors[0]
        d2 = orthonormalization(eigenvectors[1], [d1])

        return d1, d2

    def compute_loss_landscape_2d(self, data_loader, checkpoint_path=None,
                                  radius=0.5, grid=41, log_scale=True,
                                  max_batches=None, use_hessian_directions=True):
        """
        Compute 2D loss landscape around a checkpoint

        Args:
            data_loader: DataLoader for loss computation
            checkpoint_path: Path to checkpoint (if None, use current model)
            radius: Radius of exploration in parameter space
            grid: Grid size (number of points along each axis)
            log_scale: Whether to use log10 scale for loss
            max_batches: Maximum batches for loss computation
            use_hessian_directions: If True, use Hessian eigenvectors; else use random

        Returns:
            dict: Results containing landscape data
        """
        # Load checkpoint if provided
        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)

        self.model.eval()

        # Save original parameters
        orig_params = [p.detach().clone() for p in self.model.parameters()]

        # Compute directions
        if use_hessian_directions:
            d1, d2 = self.compute_directions_from_hessian(data_loader, max_batches=1)
        else:
            # Random directions
            d1 = [torch.randn_like(p) for p in orig_params]
            d2 = [torch.randn_like(p) for p in orig_params]
            # Normalize
            norm1 = sum(torch.sum(d * d) for d in d1) ** 0.5
            norm2 = sum(torch.sum(d * d) for d in d2) ** 0.5
            d1 = [d / norm1 for d in d1]
            d2 = [d / norm2 for d in d2]
            d2 = orthonormalization(d2, [d1])

        # Filter-wise normalization
        for i, p in enumerate(orig_params):
            if p.ndim > 1:  # weight matrices
                d1[i] = d1[i] / (d1[i].norm() + 1e-12) * p.norm()
                d2[i] = d2[i] / (d2[i].norm() + 1e-12) * p.norm()
            else:  # bias
                if d1[i].norm() > 0:
                    d1[i] = d1[i] / d1[i].norm() * (p.norm() + 1e-12)
                if d2[i].norm() > 0:
                    d2[i] = d2[i] / d2[i].norm() * (p.norm() + 1e-12)

        # Create grid limited to [-radius, radius]
        alphas = np.linspace(-radius, radius, grid)
        betas = np.linspace(-radius, radius, grid)
        Z = np.zeros((grid, grid))

        print(f"Computing loss landscape on {grid}x{grid} grid...")

        # Scan the grid
        for i, alpha in enumerate(alphas):
            if i % max(1, grid // 10) == 0:
                print(f"  Progress: {i}/{grid} rows")

            for j, beta in enumerate(betas):
                # Set model parameters
                for p, orig, dp1, dp2 in zip(self.model.parameters(), orig_params, d1, d2):
                    p.data.copy_(orig + alpha * dp1 + beta * dp2)

                # Compute loss
                total_loss = 0.0
                num_batches = 0

                with torch.no_grad():
                    for batch_idx, (inputs, targets) in enumerate(data_loader):
                        if max_batches is not None and batch_idx >= max_batches:
                            break

                        if not hasattr(self.params, 'pack_data') or not self.params.pack_data:
                            inputs, targets = inputs.to(self.device), targets.to(self.device)

                        u = self.model(inputs)
                        loss_data = self.loss_func.data(inputs, u, targets)
                        loss_pde = self.loss_func.pde(inputs, u, targets)
                        loss_bc = self.loss_func.bc(inputs, u, targets)
                        loss = loss_data + loss_bc + loss_pde

                        total_loss += loss.item()
                        num_batches += 1

                avg_loss = total_loss / max(num_batches, 1)
                Z[j, i] = np.log10(avg_loss + 1e-12) if log_scale else avg_loss

        # Restore original parameters
        for p, orig in zip(self.model.parameters(), orig_params):
            p.data.copy_(orig)

        results = {
            'alphas': alphas,
            'betas': betas,
            'Z': Z,
            'log_scale': log_scale,
            'radius': radius,
            'grid_size': grid,
            'use_hessian_directions': use_hessian_directions
        }

        print("Loss landscape computation completed")

        return results

    def _get_epsilon_ticks(self, radius):
        """
        Build sparse, symmetric tick locations for epsilon axes based on the radius.
        Examples: radius=0.1 -> [-0.08, -0.04, 0, 0.04, 0.08]; radius=0.5 -> [-0.4, -0.2, 0, 0.2, 0.4]; radius=2 -> [-1, 0, 1]
        """
        if radius <= 0:
            return np.array([0.0])

        if radius <= 0.2:
            step = 0.04
            max_tick = min(radius * 0.8, 0.08)
        elif radius <= 0.75:
            step = 0.2
            max_tick = min(radius * 0.8, 0.4)
        elif radius <= 1.5:
            step = 0.5
            max_tick = min(radius * 0.7, 1.0)
        else:
            step = 1.0
            max_tick = min(radius * 0.6, 2.0)

        # Align to the step and keep within radius
        max_tick = min(radius, np.floor(max_tick / step) * step)
        if max_tick < 1e-9:
            return np.array([0.0])

        ticks = np.arange(-max_tick, max_tick + step / 2, step)
        return ticks

    def _clamp_landscape_results(self, results, max_radius=None):
        """
        Optionally restrict stored landscape results to a bounded epsilon range.
        """
        if max_radius is None:
            return results

        alphas = np.array(results['alphas'])
        betas = np.array(results['betas'])

        alpha_mask = np.abs(alphas) <= max_radius + 1e-12
        beta_mask = np.abs(betas) <= max_radius + 1e-12

        if alpha_mask.all() and beta_mask.all():
            return results

        clipped = dict(results)
        clipped['alphas'] = alphas[alpha_mask]
        clipped['betas'] = betas[beta_mask]
        clipped['Z'] = results['Z'][np.ix_(beta_mask, alpha_mask)]
        clipped['radius'] = max_radius
        clipped['grid_size'] = clipped['Z'].shape[0]
        return clipped

    def plot_loss_landscape_2d(self, results, save_path=None):
        """
        Plot 2D loss landscape as contour plot

        Args:
            results: Results from compute_loss_landscape_2d
            save_path: Path to save the figure

        Returns:
            matplotlib.figure.Figure: The plot figure
        """
        results = self._clamp_landscape_results(results)

        alphas = results['alphas']
        betas = results['betas']
        Z = results['Z']
        log_scale = results['log_scale']

        A, B = np.meshgrid(alphas, betas)

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        cp = ax.contourf(A, B, Z, levels=50, cmap='plasma')
        plt.colorbar(cp, ax=ax)

        # Hide x and y axes (remove ticks and labels)
        ax.set_xlabel(r"$\varepsilon_1$", fontsize=12)
        ax.set_ylabel(r"$\varepsilon_2$", fontsize=12)

        title = '2D Loss Landscape'
        if results['use_hessian_directions']:
            title += ' (Hessian top-2 directions)'
        if log_scale:
            title += ' (log10 scale)'
        ax.set_title(title)

        # Keep axis within allowed epsilon range
        max_eps = max(np.max(np.abs(alphas)), np.max(np.abs(betas)))
        ax.set_xlim(-max_eps, max_eps)
        ax.set_ylim(-max_eps, max_eps)
        ticks = self._get_epsilon_ticks(results['radius'])
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")

        return fig

    def plot_loss_landscape_3d(self, results, save_path=None, elev=30, azim=135, cmap='RdYlBu_r'):
        """
        Plot 3D loss landscape as surface plot

        Args:
            results: Results from compute_loss_landscape_2d
            save_path: Path to save the figure
            elev: Elevation angle for 3D view
            azim: Azimuth angle for 3D view
            cmap: Colormap

        Returns:
            matplotlib.figure.Figure: The plot figure
        """
        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib.colors import Normalize

        results = self._clamp_landscape_results(results)

        alphas = results['alphas']
        betas = results['betas']
        Z = results['Z']
        log_scale = results['log_scale']

        A, B = np.meshgrid(alphas, betas)

        # Normalize colors
        norm = Normalize(vmin=np.percentile(Z, 5), vmax=np.percentile(Z, 95))

        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection='3d')

        # Main surface
        surf = ax.plot_surface(
            A, B, Z,
            rstride=1, cstride=1,
            cmap=cmap, norm=norm,
            linewidth=0, antialiased=True,
            shade=False, alpha=0.95
        )

        # Camera view
        ax.view_init(elev=elev, azim=azim)

        # Hide all axes for clean visualization (matching visualize_landscape.py)
        ax.set_axis_off()
        ax.grid(False)

        # Clamp visible range
        max_eps = max(np.max(np.abs(alphas)), np.max(np.abs(betas)))
        ax.set_xlim(-max_eps, max_eps)
        ax.set_ylim(-max_eps, max_eps)

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
            print(f"3D plot saved to {save_path}")

        return fig

    def plot_loss_landscape_3d_interactive(self, results, save_path=None, show=False):
        """
        Plot interactive 3D loss landscape using Plotly (draggable, zoomable)

        Args:
            results: Results from compute_loss_landscape_2d
            save_path: Path to save the HTML file
            show: Whether to display the plot in browser

        Returns:
            plotly.graph_objects.Figure: The interactive plot figure
        """
        try:
            import plotly.graph_objects as go
            import plotly.express as px
        except ImportError:
            print("Plotly not available. Install with: pip install plotly")
            return None

        results = self._clamp_landscape_results(results)

        alphas = results['alphas']
        betas = results['betas']
        Z = results['Z']
        log_scale = results['log_scale']
        ticks = self._get_epsilon_ticks(results['radius'])
        tick_vals = ticks.tolist()
        tick_labels = [f"{t:g}" for t in tick_vals]

        # Create meshgrid
        A, B = np.meshgrid(alphas, betas)

        # Create the 3D surface plot
        fig = go.Figure(data=[go.Surface(
            x=A,
            y=B,
            z=Z,
            colorscale='RdYlBu_r',
            opacity=0.95,
            colorbar=dict(
                title=dict(
                    text='Loss' + (' (log10)' if log_scale else ''),
                    side='right'
                ),
                tickmode='linear',
                tick0=Z.min(),
                dtick=(Z.max() - Z.min()) / 10
            ),
            hovertemplate='α: %{x:.3f}<br>β: %{y:.3f}<br>Loss: %{z:.6f}<extra></extra>'
        )])

        # Update layout for better interactivity
        title = '3D Loss Landscape (Interactive)'
        if results['use_hessian_directions']:
            title += ' - Hessian Top-2 Directions'

        max_eps = max(np.max(np.abs(alphas)), np.max(np.abs(betas)))

        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                xanchor='center',
                font=dict(size=16)
            ),
            scene=dict(
                xaxis=dict(
                    title='α (Direction 1)',
                    backgroundcolor="rgb(230, 230,230)",
                    gridcolor="white",
                    showbackground=True,
                    zerolinecolor="white",
                    showgrid=False,
                    tickmode="array",
                    tickvals=tick_vals,
                    ticktext=tick_labels,
                    range=[-max_eps, max_eps],
                ),
                yaxis=dict(
                    title='β (Direction 2)',
                    backgroundcolor="rgb(230, 230,230)",
                    gridcolor="white",
                    showbackground=True,
                    zerolinecolor="white",
                    showgrid=False,
                    tickmode="array",
                    tickvals=tick_vals,
                    ticktext=tick_labels,
                    range=[-max_eps, max_eps],
                ),
                zaxis=dict(
                    title='Loss' + (' (log10)' if log_scale else ''),
                    backgroundcolor="rgb(230, 230,230)",
                    gridcolor="white",
                    showbackground=True,
                    zerolinecolor="white",
                    showgrid=False,
                ),
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.3)
                )
            ),
            width=1000,
            height=800,
            margin=dict(l=0, r=0, b=0, t=40)
        )

        # Add annotation with statistics
        annotation_text = (
            f"Grid: {results['grid_size']}×{results['grid_size']}<br>"
            f"Radius: {results['radius']:.2f}<br>"
            f"Min Loss: {Z.min():.6f}<br>"
            f"Max Loss: {Z.max():.6f}<br>"
            f"Mean Loss: {Z.mean():.6f}"
        )

        fig.add_annotation(
            text=annotation_text,
            xref="paper", yref="paper",
            x=0.02, y=0.98,
            showarrow=False,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="black",
            borderwidth=1,
            font=dict(size=10),
            align="left"
        )

        # Add interactive control buttons
        fig.update_layout(
            updatemenus=[
                # Button group 1: XY Axes control
                dict(
                    type="buttons",
                    direction="left",
                    buttons=[
                        dict(
                            args=[{"scene.xaxis.visible": True, "scene.yaxis.visible": True}],
                            label="Show XY Axes",
                            method="relayout"
                        ),
                        dict(
                            args=[{"scene.xaxis.visible": False, "scene.yaxis.visible": False}],
                            label="Hide XY Axes",
                            method="relayout"
                        )
                    ],
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0.02,
                    xanchor="left",
                    y=0.15,
                    yanchor="bottom"
                ),
                # Button group 2: Z Axis control
                dict(
                    type="buttons",
                    direction="left",
                    buttons=[
                        dict(
                            args=[{"scene.zaxis.visible": True}],
                            label="Show Z Axis",
                            method="relayout"
                        ),
                        dict(
                            args=[{"scene.zaxis.visible": False}],
                            label="Hide Z Axis",
                            method="relayout"
                        )
                    ],
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0.02,
                    xanchor="left",
                    y=0.09,
                    yanchor="bottom"
                ),
                # Button group 3: Background shadow control
                dict(
                    type="buttons",
                    direction="left",
                    buttons=[
                        dict(
                            args=[{
                                "scene.xaxis.showbackground": True,
                                "scene.yaxis.showbackground": True,
                                "scene.zaxis.showbackground": True
                            }],
                            label="Show Background",
                            method="relayout"
                        ),
                        dict(
                            args=[{
                                "scene.xaxis.showbackground": False,
                                "scene.yaxis.showbackground": False,
                                "scene.zaxis.showbackground": False
                            }],
                            label="Hide Background",
                            method="relayout"
                        )
                    ],
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0.02,
                    xanchor="left",
                    y=0.03,
                    yanchor="bottom"
                )
            ]
        )

        # Save to HTML if path provided
        if save_path:
            # Ensure .html extension
            if not save_path.endswith('.html'):
                save_path = save_path.replace('.png', '.html')

            fig.write_html(save_path)
            print(f"Interactive 3D plot saved to {save_path}")

        # Show in browser if requested
        if show:
            fig.show()

        return fig

    def plot_loss_landscape_combined(self, results, save_path=None):
        """
        Create a combined visualization with 2D contour and 3D surface side by side

        Args:
            results: Results from compute_loss_landscape_2d
            save_path: Path to save the figure

        Returns:
            matplotlib.figure.Figure: The combined plot figure
        """
        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib.colors import Normalize

        alphas = results['alphas']
        betas = results['betas']
        Z = results['Z']
        log_scale = results['log_scale']
        ticks = self._get_epsilon_ticks(results['radius'])

        A, B = np.meshgrid(alphas, betas)

        # Create figure with 2 subplots
        fig = plt.figure(figsize=(16, 6))

        # Left: 2D Contour plot
        ax1 = fig.add_subplot(121)
        cp = ax1.contourf(A, B, Z, levels=50, cmap='plasma')
        cbar1 = plt.colorbar(cp, ax=ax1)
        # cbar1.set_label('Loss' + (' (log10)' if log_scale else ''))

        # Add contour lines
        ax1.contour(A, B, Z, levels=10, colors='white', alpha=0.3, linewidths=0.5)

        # Mark the center point
        center_idx = len(alphas) // 2
        ax1.plot(0, 0, 'r*', markersize=15, label=f'θ₀ (loss={Z[center_idx, center_idx]:.4f})')

        # Hide x and y axes (remove ticks and labels)
        ax1.set_xlabel(r"$\varepsilon_1$", fontsize=12)
        ax1.set_ylabel(r"$\varepsilon_2$", fontsize=12)
        max_eps = max(np.max(np.abs(alphas)), np.max(np.abs(betas)))
        ax1.set_xlim(-max_eps, max_eps)
        ax1.set_ylim(-max_eps, max_eps)
        ax1.set_xticks(ticks)
        ax1.set_yticks(ticks)

        ax1.set_title('2D Contour View', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Right: 3D Surface plot
        ax2 = fig.add_subplot(122, projection='3d')

        # Normalize colors
        norm = Normalize(vmin=np.percentile(Z, 5), vmax=np.percentile(Z, 95))

        surf = ax2.plot_surface(
            A, B, Z,
            rstride=1, cstride=1,
            cmap='RdYlBu_r', norm=norm,
            linewidth=0, antialiased=True,
            shade=False, alpha=0.95
        )

        # Mark the center point in 3D
        ax2.scatter([0], [0], [Z[center_idx, center_idx]],
                   color='red', s=100, marker='*',
                   label=f'θ₀')

        # Camera view
        ax2.view_init(elev=30, azim=135)
        ax2.grid(False)

        # Hide x and y axes labels and ticks (but keep axis structure)
        ax2.set_xlabel(r"$\varepsilon_1$", fontsize=12)
        ax2.set_ylabel(r"$\varepsilon_2$", fontsize=12)
        ax2.set_xlim(-max_eps, max_eps)
        ax2.set_ylim(-max_eps, max_eps)
        ax2.set_xticks(ticks)
        ax2.set_yticks(ticks)

        # Keep z-axis visible
        ax2.set_zlabel('Loss' + (' (log10)' if log_scale else ''), fontsize=11)
        ax2.set_title('3D Surface View', fontsize=14)

        # Add colorbar for 3D plot
        cbar2 = fig.colorbar(surf, ax=ax2, shrink=0.5, aspect=5)
        # cbar2.set_label('Loss' + (' (log10)' if log_scale else ''))

        # Overall title
        suptitle = 'Loss Landscape Visualization'
        if results['use_hessian_directions']:
            suptitle += ' (Hessian Top-2 Directions)'
        fig.suptitle(suptitle, fontsize=16, y=0.98)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Combined plot saved to {save_path}")

        return fig

    def plot_loss_landscape_multiview(self, results, save_path=None, single_view_path=None):
        """
        Create multiple views of the 3D landscape from different angles

        Args:
            results: Results from compute_loss_landscape_2d
            save_path: Path to save the figure
            single_view_path: Optional path to save only View 2 (elev=30°, azim=135°) as its own PDF without title/z-label

        Returns:
            matplotlib.figure.Figure: The multi-view plot figure
        """
        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib.colors import Normalize

        alphas = results['alphas']
        betas = results['betas']
        Z = results['Z']
        log_scale = results['log_scale']
        ticks = self._get_epsilon_ticks(results['radius'])

        A, B = np.meshgrid(alphas, betas)
        norm = Normalize(vmin=np.percentile(Z, 5), vmax=np.percentile(Z, 95))
        max_eps = max(np.max(np.abs(alphas)), np.max(np.abs(betas)))

        # Create figure with 4 subplots (different viewing angles)
        fig = plt.figure(figsize=(14, 12))

        # Define different viewing angles
        views = [
            (30, 45, 'View 1: elev=30°, azim=45°'),
            (30, 135, 'View 2: elev=30°, azim=135°'),
            (60, 45, 'View 3: elev=60°, azim=45°'),
            (10, 225, 'View 4: elev=10°, azim=225°')
        ]

        for idx, (elev, azim, title) in enumerate(views, 1):
            ax = fig.add_subplot(2, 2, idx, projection='3d')

            surf = ax.plot_surface(
                A, B, Z,
                rstride=1, cstride=1,
                cmap='RdYlBu_r', norm=norm,
                linewidth=0, antialiased=True,
                shade=False, alpha=0.95
            )

            ax.view_init(elev=elev, azim=azim)
            ax.grid(False)

            # Hide x and y axes labels and ticks (but keep axis structure)
            ax.set_xlabel(r"$\varepsilon_1$", fontsize=12)
            ax.set_ylabel(r"$\varepsilon_2$", fontsize=12)
            ax.set_xlim(-max_eps, max_eps)
            ax.set_ylim(-max_eps, max_eps)
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)

            # Keep z-axis visible
            ax.set_zlabel('Loss', fontsize=9)
            ax.set_title(title, fontsize=11)

            # Make ticks smaller for z-axis
            ax.tick_params(axis='z', which='major', labelsize=8)

        if single_view_path:
            # Save only View 2 (elev=30, azim=135) without title or z-label
            fig_single = plt.figure(figsize=(7, 6))
            ax_single = fig_single.add_subplot(111, projection='3d')
            ax_single.plot_surface(
                A, B, Z,
                rstride=1, cstride=1,
                cmap='RdYlBu_r', norm=norm,
                linewidth=0, antialiased=True,
                shade=False, alpha=0.95
            )
            ax_single.view_init(elev=30, azim=135)
            ax_single.grid(False)
            ax_single.set_xlabel(r"$\varepsilon_1$", fontsize=12)
            ax_single.set_ylabel(r"$\varepsilon_2$", fontsize=12)
            ax_single.set_xlim(-max_eps, max_eps)
            ax_single.set_ylim(-max_eps, max_eps)
            ax_single.set_xticks(ticks)
            ax_single.set_yticks(ticks)
            ax_single.set_zlabel('')
            ax_single.set_title('')
            fig_single.savefig(single_view_path, dpi=300, bbox_inches='tight')
            plt.close(fig_single)

        # Overall title
        suptitle = 'Loss Landscape - Multiple Viewing Angles'
        if results['use_hessian_directions']:
            suptitle += '\n(Hessian Top-2 Directions)'
        fig.suptitle(suptitle, fontsize=14, y=0.98)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Multi-view plot saved to {save_path}")

        return fig

    @classmethod
    def from_config(cls, params):
        """
        Create LossLandscapeCalculator from config parameters

        Args:
            params: Configuration parameters (YParams object)

        Returns:
            LossLandscapeCalculator: Initialized calculator
        """
        return cls(params)


class HessianCalculator_Pyhessian():
    """Class to calculate Hessian matrix properties using PyHessian methods from hessian.py"""

    def __init__(self, params, args=None):
        """
        Initialize HessianCalculator_Pyhessian
        
        Args:
            params: Configuration parameters (YParams object)
            args: Optional arguments object
        """
        self.params = params
        
        # Set device
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
        else:
            self.device = torch.device('cpu')
        
        self.params.device = self.device
        
        # Set random seed for reproducibility
        set_seed(self.params, world_size=0)  # No distributed training for Hessian
        
        # Initialize model based on config
        if self.params.model == 'fno':
            self.model = models.fno.fno(self.params).to(self.device)
        else:
            raise ValueError(f"Unsupported model type: {self.params.model}")
        
        # Set batch size parameters (similar to Trainer class)
        self.params['global_batch_size'] = self.params.batch_size
        self.params['local_batch_size'] = self.params.batch_size  # No distributed training for Hessian
        self.params['global_valid_batch_size'] = self.params.valid_batch_size
        self.params['local_valid_batch_size'] = self.params.valid_batch_size  # No distributed training
        
        # Get model parameters
        self.model_params = [p for p in self.model.parameters() if p.requires_grad]
        self.n_params = sum(p.numel() for p in self.model_params)
        
        # Initialize loss function wrapper
        self.loss_func = LossMSE(self.params, self.model)
        
        print(f"HessianCalculator_Pyhessian initialized with {self.n_params} parameters")
    
    def _create_criterion(self):
        """
        Create a criterion function compatible with PyHessian
        Returns a function that takes (outputs, targets) and returns loss
        """
        def criterion_func(outputs, targets):
            # Convert outputs to the format expected by loss_func
            # Assuming outputs is the model prediction and targets is ground truth
            # We need to create dummy inputs for the loss function
            dummy_inputs = torch.zeros_like(targets)
            
            # Compute loss using the existing loss function
            loss_data = self.loss_func.data(dummy_inputs, outputs, targets)
            loss_pde = self.loss_func.pde(dummy_inputs, outputs, targets)
            loss_bc = self.loss_func.bc(dummy_inputs, outputs, targets)
            total_loss = loss_data + loss_pde + loss_bc
            
            return total_loss
        
        return criterion_func
    
    def _prepare_data_for_pyhessian(self, data_loader, max_batches=None, use_single_batch=True):
        """
        Prepare data in the format required by PyHessian
        
        Args:
            data_loader: DataLoader for extracting data
            max_batches: Maximum number of batches to use
            use_single_batch: If True, return single batch; if False, return dataloader
            
        Returns:
            Either (inputs, targets) tuple for single batch or modified dataloader
        """
        if use_single_batch:
            # Extract first batch for single batch computation
            for batch_idx, (inputs, targets) in enumerate(data_loader):
                if not hasattr(self.params, 'pack_data') or not self.params.pack_data:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                return (inputs, targets)
        else:
            # Create a limited dataloader if max_batches is specified
            if max_batches is not None:
                limited_data = []
                for batch_idx, (inputs, targets) in enumerate(data_loader):
                    if batch_idx >= max_batches:
                        break
                    if not hasattr(self.params, 'pack_data') or not self.params.pack_data:
                        inputs, targets = inputs.to(self.device), targets.to(self.device)
                    limited_data.append((inputs, targets))
                return limited_data
            else:
                return data_loader
    
    def compute_eigenvalues(self, data_loader, top_n=1, max_iter=100, tol=1e-3, 
                           max_batches=None, use_single_batch=True):
        """
        Compute top eigenvalues using PyHessian power iteration method
        
        Args:
            data_loader: DataLoader for computation
            top_n: Number of top eigenvalues to compute
            max_iter: Maximum iterations for power iteration
            tol: Convergence tolerance
            max_batches: Maximum number of batches to process
            use_single_batch: Whether to use single batch or full dataset
            
        Returns:
            dict: Results containing eigenvalues and analysis
        """
        from utils.hessian import hessian
        
        print(f"Computing top {top_n} eigenvalues using PyHessian power iteration...")
        
        # Prepare data
        if use_single_batch:
            data = self._prepare_data_for_pyhessian(data_loader, max_batches, use_single_batch=True)
            dataloader = None
        else:
            data = None
            dataloader = self._prepare_data_for_pyhessian(data_loader, max_batches, use_single_batch=False)
        
        # Create criterion
        criterion = self._create_criterion()
        
        # Initialize PyHessian
        hessian_calc = hessian(
            model=self.model,
            criterion=criterion,
            data=data,
            dataloader=dataloader,
            cuda=torch.cuda.is_available()
        )
        
        # Compute eigenvalues
        eigenvalues, eigenvectors = hessian_calc.eigenvalues(
            maxIter=max_iter,
            tol=tol,
            top_n=top_n
        )
        
        results = {
            'eigenvalues': eigenvalues,
            'largest_eigenvalue': eigenvalues[0] if eigenvalues else 0.0,
            'top_n': top_n,
            'max_iter': max_iter,
            'tol': tol,
            'use_single_batch': use_single_batch,
            'num_params': self.n_params
        }
        
        print(f"Top {top_n} eigenvalues: {eigenvalues}")
        if eigenvalues:
            print(f"Largest eigenvalue: {eigenvalues[0]:.6f}")
        
        return results
    
    def compute_trace(self, data_loader, max_iter=100, tol=1e-3, 
                     max_batches=None, use_single_batch=True):
        """
        Compute Hessian trace using PyHessian Hutchinson's method
        
        Args:
            data_loader: DataLoader for computation
            max_iter: Maximum iterations for Hutchinson estimator
            tol: Convergence tolerance
            max_batches: Maximum number of batches to process
            use_single_batch: Whether to use single batch or full dataset
            
        Returns:
            dict: Results containing trace analysis
        """
        from utils.hessian import hessian
        
        print(f"Computing Hessian trace using PyHessian Hutchinson's method...")
        
        # Prepare data
        if use_single_batch:
            data = self._prepare_data_for_pyhessian(data_loader, max_batches, use_single_batch=True)
            dataloader = None
        else:
            data = None
            dataloader = self._prepare_data_for_pyhessian(data_loader, max_batches, use_single_batch=False)
        
        # Create criterion
        criterion = self._create_criterion()
        
        # Initialize PyHessian
        hessian_calc = hessian(
            model=self.model,
            criterion=criterion,
            data=data,
            dataloader=dataloader,
            cuda=torch.cuda.is_available()
        )
        
        # Compute trace
        trace_vhv = hessian_calc.trace(maxIter=max_iter, tol=tol)
        
        # Calculate statistics
        trace_mean = np.mean(trace_vhv)
        trace_std = np.std(trace_vhv)
        avg_curvature = trace_mean / self.n_params
        
        results = {
            'trace': trace_mean,
            'trace_std': trace_std,
            'trace_samples': trace_vhv,
            'avg_curvature': avg_curvature,
            'max_iter': max_iter,
            'tol': tol,
            'use_single_batch': use_single_batch,
            'num_params': self.n_params,
            'convergence_samples': len(trace_vhv)
        }
        
        print(f"Hessian trace: {trace_mean:.6f} ± {trace_std:.6f}")
        print(f"Average curvature: {avg_curvature:.6f}")
        print(f"Converged after {len(trace_vhv)} samples")
        
        return results
    
    def compute_eigenvalue_density(self, data_loader, iter=100, n_v=1,
                                  max_batches=None, use_single_batch=True):
        """
        Compute eigenvalue density using PyHessian Stochastic Lanczos Quadrature (SLQ)
        
        Args:
            data_loader: DataLoader for computation
            iter: Number of Lanczos iterations
            n_v: Number of SLQ runs
            max_batches: Maximum number of batches to process
            use_single_batch: Whether to use single batch or full dataset
            
        Returns:
            dict: Results containing eigenvalue density analysis
        """
        from utils.hessian import hessian
        
        print(f"Computing eigenvalue density using PyHessian SLQ method...")
        print(f"Lanczos iterations: {iter}, SLQ runs: {n_v}")
        
        # Prepare data
        if use_single_batch:
            data = self._prepare_data_for_pyhessian(data_loader, max_batches, use_single_batch=True)
            dataloader = None
        else:
            data = None
            dataloader = self._prepare_data_for_pyhessian(data_loader, max_batches, use_single_batch=False)
        
        # Create criterion
        criterion = self._create_criterion()
        
        # Initialize PyHessian
        hessian_calc = hessian(
            model=self.model,
            criterion=criterion,
            data=data,
            dataloader=dataloader,
            cuda=torch.cuda.is_available()
        )
        
        # Compute eigenvalue density
        eigen_list_full, weight_list_full = hessian_calc.density(iter=iter, n_v=n_v)
        
        # Process results
        all_eigenvals = []
        all_weights = []
        for eigenvals, weights in zip(eigen_list_full, weight_list_full):
            all_eigenvals.extend(eigenvals)
            all_weights.extend(weights)
        
        # Convert to numpy arrays
        all_eigenvals = np.array(all_eigenvals)
        all_weights = np.array(all_weights)
        
        # Calculate statistics
        eigenval_stats = {
            'mean': np.mean(all_eigenvals),
            'std': np.std(all_eigenvals),
            'min': np.min(all_eigenvals),
            'max': np.max(all_eigenvals),
            'median': np.median(all_eigenvals),
            'num_negative': np.sum(all_eigenvals < 0),
            'num_positive': np.sum(all_eigenvals > 0),
            'num_total': len(all_eigenvals)
        }
        
        results = {
            'eigen_list_full': eigen_list_full,
            'weight_list_full': weight_list_full,
            'all_eigenvals': all_eigenvals,
            'all_weights': all_weights,
            'eigenval_stats': eigenval_stats,
            'lanczos_iter': iter,
            'slq_runs': n_v,
            'use_single_batch': use_single_batch,
            'num_params': self.n_params
        }
        
        print(f"Eigenvalue statistics:")
        print(f"  Mean: {eigenval_stats['mean']:.6f}")
        print(f"  Std: {eigenval_stats['std']:.6f}")
        print(f"  Min: {eigenval_stats['min']:.6f}")
        print(f"  Max: {eigenval_stats['max']:.6f}")
        print(f"  Negative eigenvalues: {eigenval_stats['num_negative']}/{eigenval_stats['num_total']}")
        
        return results
    
    def analyze_hessian_properties(self, data_loader, method='all', layerwise=False, **kwargs):
        """
        Comprehensive Hessian analysis using PyHessian methods
        
        Args:
            data_loader: DataLoader for analysis
            method: 'eigenvalues', 'trace', 'density', or 'all'
            layerwise: If True, compute Hessian properties for each layer
            **kwargs: Additional arguments for specific methods
            
        Returns:
            dict: Comprehensive analysis results
        """
        results = {}
        
        if layerwise:
            print("Performing layerwise Hessian analysis using PyHessian...")
            # Use the layerwise method
            layerwise_results = self.compute_layerwise_hessian(data_loader, method=method, **kwargs)
            results['layerwise'] = layerwise_results
            
            # Print layerwise summary
            print("\n" + "="*60)
            print("LAYERWISE PYHESSIAN ANALYSIS SUMMARY")
            print("="*60)
            summary = layerwise_results['summary']
            print(f"Total layers analyzed: {summary['total_layers']}")
            print(f"Successful layers: {summary['successful_layers']}")
            print(f"Failed layers: {summary['failed_layers']}")
            print(f"Layer names: {summary['layer_names']}")
            print("="*60)
            
        else:
            print("Performing comprehensive Hessian analysis using PyHessian...")
            
            if method in ['eigenvalues', 'all']:
                print("\n1. Computing eigenvalues...")
                eigenval_kwargs = {k: v for k, v in kwargs.items() if k in ['top_n', 'max_iter', 'tol', 'max_batches', 'use_single_batch']}
                eigenval_results = self.compute_eigenvalues(data_loader, **eigenval_kwargs)
                results['eigenvalues'] = eigenval_results
            
            if method in ['trace', 'all']:
                print("\n2. Computing trace...")
                trace_kwargs = {k: v for k, v in kwargs.items() if k in ['max_iter', 'tol', 'max_batches', 'use_single_batch']}
                trace_results = self.compute_trace(data_loader, **trace_kwargs)
                results['trace'] = trace_results
            
            if method in ['density', 'all']:
                print("\n3. Computing eigenvalue density...")
                density_kwargs = {k: v for k, v in kwargs.items() if k in ['iter', 'n_v', 'max_batches', 'use_single_batch']}
                density_results = self.compute_eigenvalue_density(data_loader, **density_kwargs)
                results['density'] = density_results
            
            # Summary statistics
            print("\n" + "="*60)
            print("PYHESSIAN ANALYSIS SUMMARY")
            print("="*60)
            
            if 'eigenvalues' in results:
                eigenvals = results['eigenvalues']['eigenvalues']
                if eigenvals:
                    print(f"Largest eigenvalue: {eigenvals[0]:.6f}")
                    print(f"Top {len(eigenvals)} eigenvalues: {eigenvals}")
            
            if 'trace' in results:
                trace_val = results['trace']['trace']
                avg_curv = results['trace']['avg_curvature']
                print(f"Hessian trace: {trace_val:.6f}")
                print(f"Average curvature: {avg_curv:.6f}")
            
            if 'density' in results:
                stats = results['density']['eigenval_stats']
                print(f"Eigenvalue density stats:")
                print(f"  Range: [{stats['min']:.6f}, {stats['max']:.6f}]")
                print(f"  Mean: {stats['mean']:.6f}")
                print(f"  Negative/Total: {stats['num_negative']}/{stats['num_total']}")
            
            print("="*60)
        
        return results
    
    def compute_sharpness_metrics(self, data_loader, layerwise=False, **kwargs):
        """
        Compute sharpness metrics using both eigenvalues and trace
        
        Args:
            data_loader: DataLoader for computation
            layerwise: If True, compute sharpness metrics for each layer
            **kwargs: Additional arguments
            
        Returns:
            dict: Sharpness analysis results
        """
        if layerwise:
            print("Computing layerwise sharpness metrics using PyHessian...")
        else:
            print("Computing sharpness metrics using PyHessian...")
        
        # Compute eigenvalues and trace
        results = self.analyze_hessian_properties(data_loader, method='all', layerwise=layerwise, **kwargs)
        
        # Extract sharpness metrics
        sharpness_results = {
            'method': 'PyHessian',
            'num_params': self.n_params,
            'layerwise': layerwise
        }
        
        if layerwise and 'layerwise' in results:
            # Process layerwise results
            layerwise_sharpness = {}
            layerwise_data = results['layerwise']['layerwise_results']
            
            for layer_name, layer_results in layerwise_data.items():
                if 'error' in layer_results:
                    layerwise_sharpness[layer_name] = {'error': layer_results['error']}
                    continue
                
                layer_sharpness = {
                    'layer_name': layer_name,
                    'layer_type': layer_results.get('layer_type', 'unknown'),
                    'layer_params': layer_results.get('layer_params', 0)
                }
                
                # Extract eigenvalue-based sharpness
                if 'eigenvalues' in layer_results and 'error' not in layer_results['eigenvalues']:
                    eigenvals = layer_results['eigenvalues']['eigenvalues']
                    if eigenvals:
                        layer_sharpness['largest_eigenvalue'] = eigenvals[0]
                        layer_sharpness['eigenvalue_sharpness'] = eigenvals[0]
                
                # Extract trace-based sharpness
                if 'trace' in layer_results and 'error' not in layer_results['trace']:
                    trace_val = layer_results['trace']['trace']
                    layer_sharpness['trace'] = trace_val
                    layer_sharpness['trace_sharpness'] = trace_val
                    layer_sharpness['avg_curvature'] = layer_results['trace']['avg_curvature']
                
                # Extract density-based metrics
                if 'density' in layer_results and 'error' not in layer_results['density']:
                    stats = layer_results['density']['eigenval_stats']
                    if 'error' not in stats:
                        layer_sharpness['eigenval_max'] = stats['max']
                        layer_sharpness['eigenval_min'] = stats['min']
                        layer_sharpness['eigenval_mean'] = stats['mean']
                        layer_sharpness['condition_number_approx'] = stats['max'] / max(abs(stats['min']), 1e-10)
                        layer_sharpness['negative_eigenvals'] = stats['num_negative']
                        layer_sharpness['total_eigenvals'] = stats['num_total']
                
                layerwise_sharpness[layer_name] = layer_sharpness
            
            sharpness_results['layerwise_sharpness'] = layerwise_sharpness
            sharpness_results['summary'] = results['layerwise']['summary']
            
            # Print layerwise summary
            print(f"Layerwise sharpness metrics computed for {len(layerwise_sharpness)} layers:")
            for layer_name, layer_sharpness in layerwise_sharpness.items():
                if 'error' in layer_sharpness:
                    print(f"  {layer_name}: Error - {layer_sharpness['error']}")
                else:
                    metrics = []
                    if 'largest_eigenvalue' in layer_sharpness:
                        metrics.append(f"λ_max={layer_sharpness['largest_eigenvalue']:.4f}")
                    if 'trace' in layer_sharpness:
                        metrics.append(f"tr={layer_sharpness['trace']:.4f}")
                    print(f"  {layer_name}: {', '.join(metrics) if metrics else 'No valid metrics'}")
        
        else:
            # Process standard (full model) results
            if 'eigenvalues' in results:
                eigenvals = results['eigenvalues']['eigenvalues']
                if eigenvals:
                    sharpness_results['largest_eigenvalue'] = eigenvals[0]
                    sharpness_results['eigenvalue_sharpness'] = eigenvals[0]
            
            if 'trace' in results:
                trace_val = results['trace']['trace']
                sharpness_results['trace'] = trace_val
                sharpness_results['trace_sharpness'] = trace_val
                sharpness_results['avg_curvature'] = results['trace']['avg_curvature']
            
            if 'density' in results:
                stats = results['density']['eigenval_stats']
                sharpness_results['eigenval_max'] = stats['max']
                sharpness_results['eigenval_min'] = stats['min']
                sharpness_results['eigenval_mean'] = stats['mean']
                sharpness_results['condition_number_approx'] = stats['max'] / max(abs(stats['min']), 1e-10)
                sharpness_results['negative_eigenvals'] = stats['num_negative']
                sharpness_results['total_eigenvals'] = stats['num_total']
            
            print(f"Sharpness metrics computed:")
            if 'largest_eigenvalue' in sharpness_results:
                print(f"  Largest eigenvalue: {sharpness_results['largest_eigenvalue']:.6f}")
            if 'trace' in sharpness_results:
                print(f"  Trace: {sharpness_results['trace']:.6f}")
        
        # Add full results for detailed analysis
        sharpness_results['detailed_results'] = results
        
        return sharpness_results
    
    def save_layerwise_info_to_file(self, output_dir, results=None):
        """
        Save detailed layerwise information to text files for analysis
        
        Args:
            output_dir: Directory to save the files
            results: Optional layerwise results to include Hessian metrics
        """
        import os
        
        # Save layer dimensions and architecture info
        layer_info_file = os.path.join(output_dir, "layer_dimensions_info.txt")
        
        with open(layer_info_file, 'w') as f:
            f.write("LAYER DIMENSIONS AND ARCHITECTURE INFO\n")
            f.write("="*80 + "\n")
            
            # Get layer information
            layer_info = self._get_layer_names()
            
            f.write(f"Total layers: {len(layer_info)}\n")
            f.write(f"Total model parameters: {self.n_params:,}\n\n")
            
            f.write("Layer Details:\n")
            f.write("-" * 100 + "\n")
            f.write(f"{'Layer Name':<25} {'Type':<20} {'Parameters':<15} {'Percentage':<12} {'Shape Info'}\n")
            f.write("-" * 100 + "\n")
            
            total_params = self.n_params
            
            for layer_name, layer_module in layer_info:
                layer_params = sum(p.numel() for p in layer_module.parameters() if p.requires_grad)
                param_percentage = (layer_params / max(total_params, 1)) * 100
                layer_type = type(layer_module).__name__
                
                # Get shape information
                shapes = []
                for name, param in layer_module.named_parameters():
                    if param.requires_grad:
                        shapes.append(f"{name}: {list(param.shape)}")
                shape_info = "; ".join(shapes) if shapes else "No trainable params"
                
                f.write(f"{layer_name:<25} {layer_type:<20} {layer_params:<15,} {param_percentage:<11.2f}% {shape_info}\n")
            
            f.write("-" * 100 + "\n")
        
        print(f"Layer dimensions info saved to: {layer_info_file}")
        
        # Save layerwise Hessian metrics if available
        if results and 'layerwise' in results:
            self._save_layerwise_hessian_summary(output_dir, results['layerwise'])
    
    def _save_layerwise_hessian_summary(self, output_dir, layerwise_results):
        """
        Save layerwise Hessian metrics to CSV and detailed text files
        
        Args:
            output_dir: Directory to save the files
            layerwise_results: Layerwise analysis results
        """
        import csv
        import os
        
        # Save to CSV for easy analysis
        csv_file = os.path.join(output_dir, "layerwise_hessian_summary.csv")
        
        # Define CSV headers
        headers = [
            'layer_name', 'layer_type', 'layer_params', 'param_percentage',
            'largest_eigenvalue', 'trace', 'trace_std', 'avg_curvature',
            'eigenval_max', 'eigenval_min', 'eigenval_mean',
            'negative_eigenvals', 'total_eigenvals', 'condition_number_approx',
            'has_error', 'error_message'
        ]
        
        layerwise_data = layerwise_results.get('layerwise_results', {})
        
        # Calculate total parameters for percentage calculation
        total_params = sum(layer_info.get('layer_params', 0) 
                          for layer_info in layerwise_data.values() 
                          if 'error' not in layer_info)
        
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            
            for layer_name, layer_info in layerwise_data.items():
                # Handle error cases
                if 'error' in layer_info:
                    row = [
                        layer_name, 
                        layer_info.get('layer_type', 'unknown'),
                        layer_info.get('layer_params', 0),
                        0.0,  # param_percentage
                        '', '', '', '', '', '', '', '', '', '',  # empty metrics
                        True,  # has_error
                        layer_info['error']  # error_message
                    ]
                    writer.writerow(row)
                    continue
                
                # Extract layer metadata
                layer_params = layer_info.get('layer_params', 0)
                param_percentage = (layer_params / max(total_params, 1)) * 100
                
                # Extract eigenvalue metrics
                largest_eigenvalue = ''
                if 'eigenvalues' in layer_info and 'error' not in layer_info['eigenvalues']:
                    eigenvals = layer_info['eigenvalues'].get('eigenvalues', [])
                    if eigenvals:
                        largest_eigenvalue = eigenvals[0]
                
                # Extract trace metrics
                trace = trace_std = avg_curvature = '', '', ''
                if 'trace' in layer_info and 'error' not in layer_info['trace']:
                    trace = layer_info['trace'].get('trace', '')
                    trace_std = layer_info['trace'].get('trace_std', '')
                    avg_curvature = layer_info['trace'].get('avg_curvature', '')
                
                # Extract density metrics (skip if disabled)
                eigenval_max = eigenval_min = eigenval_mean = '', '', ''
                negative_eigenvals = total_eigenvals = condition_number_approx = '', '', ''
                if ('density' in layer_info and 
                    'error' not in layer_info['density'] and 
                    not layer_info['density'].get('disabled', False)):
                    stats = layer_info['density'].get('eigenval_stats', {})
                    if 'error' not in stats:
                        eigenval_max = stats.get('max', '')
                        eigenval_min = stats.get('min', '')
                        eigenval_mean = stats.get('mean', '')
                        negative_eigenvals = stats.get('num_negative', '')
                        total_eigenvals = stats.get('num_total', '')
                        if eigenval_max and eigenval_min:
                            condition_number_approx = eigenval_max / max(abs(eigenval_min), 1e-10)
                
                # Create row
                row = [
                    layer_name,
                    layer_info.get('layer_type', 'unknown'),
                    layer_params,
                    f"{param_percentage:.2f}",
                    largest_eigenvalue,
                    trace,
                    trace_std,
                    avg_curvature,
                    eigenval_max,
                    eigenval_min,
                    eigenval_mean,
                    negative_eigenvals,
                    total_eigenvals,
                    condition_number_approx,
                    False,  # has_error
                    ''      # error_message
                ]
                
                writer.writerow(row)
        
        print(f"Layerwise Hessian summary saved to: {csv_file}")
        
        # Save detailed text summary
        text_file = os.path.join(output_dir, "layerwise_hessian_detailed.txt")
        
        with open(text_file, 'w') as f:
            f.write("LAYERWISE HESSIAN ANALYSIS DETAILED RESULTS\n")
            f.write("="*80 + "\n")
            
            summary = layerwise_results.get('summary', {})
            f.write(f"Total layers analyzed: {summary.get('total_layers', 0)}\n")
            f.write(f"Successful layers: {summary.get('successful_layers', 0)}\n")
            f.write(f"Failed layers: {summary.get('failed_layers', 0)}\n")
            f.write(f"Analysis method: {summary.get('method', 'unknown')}\n\n")
            
            for layer_name, layer_info in layerwise_data.items():
                f.write(f"Layer: {layer_name}\n")
                f.write("-" * 40 + "\n")
                
                if 'error' in layer_info:
                    f.write(f"  Error: {layer_info['error']}\n\n")
                    continue
                
                f.write(f"  Type: {layer_info.get('layer_type', 'unknown')}\n")
                f.write(f"  Parameters: {layer_info.get('layer_params', 0):,}\n")
                
                # Eigenvalue info
                if 'eigenvalues' in layer_info and 'error' not in layer_info['eigenvalues']:
                    eigenvals = layer_info['eigenvalues'].get('eigenvalues', [])
                    if eigenvals:
                        f.write(f"  Largest eigenvalue: {eigenvals[0]:.8f}\n")
                        if len(eigenvals) > 1:
                            f.write(f"  Top eigenvalues: {eigenvals}\n")
                
                # Trace info
                if 'trace' in layer_info and 'error' not in layer_info['trace']:
                    trace_data = layer_info['trace']
                    f.write(f"  Trace: {trace_data.get('trace', 'N/A'):.8f}\n")
                    f.write(f"  Trace std: {trace_data.get('trace_std', 'N/A'):.8f}\n")
                    f.write(f"  Avg curvature: {trace_data.get('avg_curvature', 'N/A'):.8f}\n")
                
                # Density info (if available and not disabled)
                if ('density' in layer_info and 
                    'error' not in layer_info['density'] and 
                    not layer_info['density'].get('disabled', False)):
                    stats = layer_info['density'].get('eigenval_stats', {})
                    if 'error' not in stats:
                        f.write(f"  Eigenval range: [{stats.get('min', 'N/A'):.8f}, {stats.get('max', 'N/A'):.8f}]\n")
                        f.write(f"  Eigenval mean: {stats.get('mean', 'N/A'):.8f}\n")
                        f.write(f"  Negative eigenvals: {stats.get('num_negative', 'N/A')}/{stats.get('num_total', 'N/A')}\n")
                
                f.write("\n")
        
        print(f"Detailed layerwise analysis saved to: {text_file}")
    
    def load_model(self, checkpoint_path):
        """
        Load model weights from checkpoint
        
        Args:
            checkpoint_path: Path to the checkpoint file
        """
        print(f"Loading model weights from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu')
        
        try:
            self.model.load_state_dict(checkpoint['model_state'])
        except:
            # Handle DistributedDataParallel models
            new_state_dict = OrderedDict()
            for key, val in checkpoint['model_state'].items():
                if key.startswith('module.'):
                    name = key[7:]  # Remove 'module.' prefix
                else:
                    name = key
                new_state_dict[name] = val
            self.model.load_state_dict(new_state_dict)
        
        print("Model weights loaded successfully")
    
    def restore_checkpoint(self, checkpoint_path):
        """
        Restore full checkpoint including model state and metadata
        
        Args:
            checkpoint_path: Path to the checkpoint file
        
        Returns:
            dict: Checkpoint metadata (epoch, iters, etc.)
        """
        print(f"Restoring checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu')
        
        try:
            self.model.load_state_dict(checkpoint['model_state'])
        except:
            # Handle DistributedDataParallel models
            new_state_dict = OrderedDict()
            for key, val in checkpoint['model_state'].items():
                if key.startswith('module.'):
                    name = key[7:]  # Remove 'module.' prefix
                else:
                    name = key
                new_state_dict[name] = val
            self.model.load_state_dict(new_state_dict)
        
        # Return checkpoint metadata
        metadata = {
            'epoch': checkpoint.get('epoch', 0),
            'iters': checkpoint.get('iters', 0)
        }
        
        print(f"Checkpoint restored successfully (epoch: {metadata['epoch']}, iters: {metadata['iters']})")
        return metadata
    
    def _get_layer_names(self):
        """
        Get meaningful layer names for Hessian analysis
        
        Returns:
            list: List of (layer_name, module) tuples for analysis
        """
        layer_info = []
        
        # For FNO model, analyze key components
        if hasattr(self.model, 'fc0'):
            layer_info.append(('input_projection', self.model.fc0))
        
        if hasattr(self.model, 'sp_convs'):
            for i, conv in enumerate(self.model.sp_convs):
                layer_info.append((f'spectral_conv_{i}', conv))
        
        if hasattr(self.model, 'ws'):
            for i, w in enumerate(self.model.ws):
                layer_info.append((f'skip_connection_{i}', w))
        
        if hasattr(self.model, 'fc1'):
            layer_info.append(('hidden_projection', self.model.fc1))
            
        if hasattr(self.model, 'fc2'):
            layer_info.append(('output_projection', self.model.fc2))
        
        # For other model types, add general approach
        if not layer_info:
            for name, module in self.model.named_modules():
                if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                    layer_info.append((name, module))
        
        return layer_info
    
    def _create_layerwise_model(self, target_layer_name):
        """
        Create a model wrapper that only has gradients for the target layer
        
        Args:
            target_layer_name: Name of the layer to analyze
            
        Returns:
            nn.Module: Model with only target layer parameters requiring gradients
        """
        # Create a copy of the full model but freeze all layers except target
        class LayerwiseModel(nn.Module):
            def __init__(self, base_model, target_layer_name):
                super().__init__()
                self.base_model = base_model
                self.target_layer_name = target_layer_name
                
                # Freeze all parameters first
                for param in self.base_model.parameters():
                    param.requires_grad = False
                
                # Enable gradients only for target layer
                target_module = self._get_target_module()
                if target_module is not None:
                    for param in target_module.parameters():
                        param.requires_grad = True
                
            def _get_target_module(self):
                """Get the target module based on layer name"""
                if hasattr(self.base_model, 'fc0') and self.target_layer_name == 'input_projection':
                    return self.base_model.fc0
                elif hasattr(self.base_model, 'sp_convs') and 'spectral_conv_' in self.target_layer_name:
                    idx = int(self.target_layer_name.split('_')[-1])
                    if idx < len(self.base_model.sp_convs):
                        return self.base_model.sp_convs[idx]
                elif hasattr(self.base_model, 'ws') and 'skip_connection_' in self.target_layer_name:
                    idx = int(self.target_layer_name.split('_')[-1])
                    if idx < len(self.base_model.ws):
                        return self.base_model.ws[idx]
                elif hasattr(self.base_model, 'fc1') and self.target_layer_name == 'hidden_projection':
                    return self.base_model.fc1
                elif hasattr(self.base_model, 'fc2') and self.target_layer_name == 'output_projection':
                    return self.base_model.fc2
                return None
                
            def forward(self, x):
                # Return the full model output (but only target layer has gradients)
                return self.base_model(x)
            
            def parameters(self):
                """Only return parameters that require gradients (target layer only)"""
                for param in self.base_model.parameters():
                    if param.requires_grad:
                        yield param
        
        return LayerwiseModel(self.model, target_layer_name)
    
    def compute_layerwise_hessian(self, data_loader, method='eigenvalues', max_batches=None, 
                                 use_single_batch=True, **kwargs):
        """
        Compute Hessian properties for each layer of the model
        
        Args:
            data_loader: DataLoader for computation
            method: 'eigenvalues', 'trace', 'density', or 'all'
            max_batches: Maximum number of batches to process
            use_single_batch: Whether to use single batch
            **kwargs: Additional arguments for specific Hessian methods
            
        Returns:
            dict: Layerwise Hessian analysis results
        """
        from utils.hessian import hessian
        
        print("Computing layerwise Hessian analysis...")
        print(f"Method: {method}")
        
        # Get layer information
        layer_info = self._get_layer_names()
        print(f"Analyzing {len(layer_info)} layers: {[name for name, _ in layer_info]}")
        
        layerwise_results = {}
        
        # Store original model state
        original_model = self.model
        
        for layer_name, layer_module in layer_info:
            print(f"\nAnalyzing layer: {layer_name}")
            
            try:
                # Create a criterion that focuses on this layer's parameters
                def layer_criterion(outputs, targets):
                    # Use the same loss as the full model
                    dummy_inputs = torch.zeros_like(targets)
                    loss_data = self.loss_func.data(dummy_inputs, outputs, targets)
                    loss_pde = self.loss_func.pde(dummy_inputs, outputs, targets)
                    loss_bc = self.loss_func.bc(dummy_inputs, outputs, targets)
                    return loss_data + loss_pde + loss_bc
                
                # Prepare data
                if use_single_batch:
                    data = self._prepare_data_for_pyhessian(data_loader, max_batches, use_single_batch=True)
                    dataloader = None
                else:
                    data = None
                    dataloader = self._prepare_data_for_pyhessian(data_loader, max_batches, use_single_batch=False)
                
                # Create truncated model for this layer
                layer_model = self._create_layerwise_model(layer_name)
                layer_model.eval()
                
                # Initialize PyHessian for this layer
                hessian_calc = hessian(
                    model=layer_model,
                    criterion=layer_criterion,
                    data=data,
                    dataloader=dataloader,
                    cuda=torch.cuda.is_available()
                )
                
                # Compute Hessian properties based on method
                layer_results = {}
                
                if method in ['eigenvalues', 'all']:
                    print(f"  Computing eigenvalues for {layer_name}...")
                    try:
                        top_n = kwargs.get('top_n', 1)
                        max_iter = kwargs.get('max_iter', 50)  # Reduced for layer analysis
                        tol = kwargs.get('tol', 1e-2)  # Relaxed tolerance
                        
                        eigenvalues, _ = hessian_calc.eigenvalues(
                            maxIter=max_iter, tol=tol, top_n=top_n
                        )
                        
                        layer_results['eigenvalues'] = {
                            'eigenvalues': eigenvalues,
                            'largest_eigenvalue': eigenvalues[0] if eigenvalues else 0.0,
                            'top_n': top_n,
                            'max_iter': max_iter,
                            'tol': tol
                        }
                        print(f"    Largest eigenvalue: {eigenvalues[0] if eigenvalues else 0.0:.6f}")
                        
                    except Exception as e:
                        print(f"    Warning: Eigenvalue computation failed for {layer_name}: {e}")
                        layer_results['eigenvalues'] = {'error': str(e)}
                
                if method in ['trace', 'all']:
                    print(f"  Computing trace for {layer_name}...")
                    try:
                        max_iter = kwargs.get('max_iter', 50)  # Reduced for layer analysis
                        tol = kwargs.get('tol', 1e-2)  # Relaxed tolerance
                        
                        trace_vhv = hessian_calc.trace(maxIter=max_iter, tol=tol)
                        trace_mean = np.mean(trace_vhv)
                        trace_std = np.std(trace_vhv)
                        
                        # Get layer parameter count
                        layer_params = sum(p.numel() for p in layer_module.parameters() if p.requires_grad)
                        avg_curvature = trace_mean / max(layer_params, 1)
                        
                        layer_results['trace'] = {
                            'trace': trace_mean,
                            'trace_std': trace_std,
                            'avg_curvature': avg_curvature,
                            'layer_params': layer_params,
                            'convergence_samples': len(trace_vhv)
                        }
                        print(f"    Trace: {trace_mean:.6f} ± {trace_std:.6f}")
                        
                    except Exception as e:
                        print(f"    Warning: Trace computation failed for {layer_name}: {e}")
                        layer_results['trace'] = {'error': str(e)}
                
                if method in ['density', 'all']:
                    print(f"  Computing eigenvalue density for {layer_name}...")
                    try:
                        iter_count = kwargs.get('iter', 50)  # Reduced for layer analysis
                        n_v = kwargs.get('n_v', 1)
                        
                        eigen_list_full, weight_list_full = hessian_calc.density(iter=iter_count, n_v=n_v)
                        
                        # Process results
                        all_eigenvals = []
                        for eigenvals in eigen_list_full:
                            all_eigenvals.extend(eigenvals)
                        
                        if all_eigenvals:
                            all_eigenvals = np.array(all_eigenvals)
                            eigenval_stats = {
                                'mean': np.mean(all_eigenvals),
                                'std': np.std(all_eigenvals),
                                'min': np.min(all_eigenvals),
                                'max': np.max(all_eigenvals),
                                'num_negative': np.sum(all_eigenvals < 0),
                                'num_total': len(all_eigenvals)
                            }
                        else:
                            eigenval_stats = {'error': 'No eigenvalues computed'}
                        
                        layer_results['density'] = {
                            'eigenval_stats': eigenval_stats,
                            'lanczos_iter': iter_count,
                            'slq_runs': n_v
                        }
                        
                        if 'error' not in eigenval_stats:
                            print(f"    Eigenvalue range: [{eigenval_stats['min']:.6f}, {eigenval_stats['max']:.6f}]")
                        
                    except Exception as e:
                        print(f"    Warning: Density computation failed for {layer_name}: {e}")
                        layer_results['density'] = {'error': str(e)}
                
                # Add layer metadata
                layer_results['layer_name'] = layer_name
                layer_results['layer_type'] = type(layer_module).__name__
                layer_results['layer_params'] = sum(p.numel() for p in layer_module.parameters() if p.requires_grad)
                
                layerwise_results[layer_name] = layer_results
                
            except Exception as e:
                print(f"  Error analyzing layer {layer_name}: {e}")
                layerwise_results[layer_name] = {
                    'layer_name': layer_name,
                    'error': str(e)
                }
        
        # Restore original model and all parameter gradients
        self.model = original_model
        # Ensure all parameters have gradients enabled again
        for param in self.model.parameters():
            param.requires_grad = True
        
        # Add summary statistics
        successful_layers = [name for name, results in layerwise_results.items() if 'error' not in results]
        
        summary = {
            'total_layers': len(layer_info),
            'successful_layers': len(successful_layers),
            'failed_layers': len(layer_info) - len(successful_layers),
            'method': method,
            'layer_names': [name for name, _ in layer_info]
        }
        
        print(f"\nLayerwise analysis complete:")
        print(f"  Total layers: {summary['total_layers']}")
        print(f"  Successful: {summary['successful_layers']}")
        print(f"  Failed: {summary['failed_layers']}")
        
        return {
            'summary': summary,
            'layerwise_results': layerwise_results
        }
    
    def print_model_architecture_summary(self):
        """
        Print a summary of the model architecture with parameter counts
        """
        print("\n" + "="*60)
        print("MODEL ARCHITECTURE SUMMARY (PyHessian)")
        print("="*60)
        
        total_params = 0
        trainable_params = 0
        
        for name, param in self.model.named_parameters():
            num_params = param.numel()
            total_params += num_params
            if param.requires_grad:
                trainable_params += num_params
                status = "Trainable"
            else:
                status = "Frozen"
            
            print(f"{name:<50} {str(param.shape):<20} {num_params:<10,} {status}")
        
        print("="*60)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Non-trainable parameters: {total_params - trainable_params:,}")
        print("="*60)
    
    @classmethod
    def from_config(cls, params, checkpoint_path=None):
        """
        Create HessianCalculator_Pyhessian from config parameters and optionally load checkpoint
        
        Args:
            params: Configuration parameters (YParams object)
            checkpoint_path: Optional path to model checkpoint
            
        Returns:
            HessianCalculator_Pyhessian: Initialized calculator
        """
        # Create calculator with parameters
        calc = cls(params)
        
        # Load checkpoint if specified
        if checkpoint_path is not None:
            calc.load_model(checkpoint_path)
        
        return calc

class LossLandscape3D():
    """Class to calculate 3D loss landscape around a model checkpoint"""

    def __init__(self, params, args=None):
        """
        Initialize LossLandscape3D calculator

        Args:
            params: Configuration parameters (YParams object)
            args: Optional arguments object
        """
        self.params = params

        # Set device
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
        else:
            self.device = torch.device('cpu')

        self.params.device = self.device

        # Set random seed for reproducibility
        set_seed(self.params, world_size=0)

        # Initialize model
        if self.params.model == 'fno':
            self.model = models.fno.fno(self.params).to(self.device)
        else:
            raise ValueError(f"Unsupported model type: {self.params.model}")

        # Set batch size parameters
        self.params['global_batch_size'] = self.params.batch_size
        self.params['local_batch_size'] = self.params.batch_size
        self.params['global_valid_batch_size'] = self.params.valid_batch_size
        self.params['local_valid_batch_size'] = self.params.valid_batch_size

        # Initialize loss function
        if self.params.loss_func == "mse":
            self.loss_func = LossMSE(self.params, self.model)
        else:
            raise ValueError(f"Unsupported loss function: {self.params.loss_func}")

        # Get model parameters
        self.model_params = [p for p in self.model.parameters() if p.requires_grad]
        self.n_params = sum(p.numel() for p in self.model_params)

        # Store original parameters
        self.original_params = None

        # Random directions for landscape exploration
        self.direction1 = None
        self.direction2 = None

        print(f"LossLandscape3D initialized with {self.n_params} parameters")

    def _get_parameter_vector(self):
        """Extract parameter vector from model"""
        return torch.cat([p.data.reshape(-1) for p in self.model.parameters() if p.requires_grad])

    def _set_parameter_vector(self, param_vector):
        """Set model parameters from vector"""
        idx = 0
        for p in self.model.parameters():
            if p.requires_grad:
                param_size = p.numel()
                p.data.copy_(param_vector[idx:idx+param_size].view(p.shape))
                idx += param_size

    def load_model(self, checkpoint_path):
        """
        Load model from checkpoint and store original parameters

        Args:
            checkpoint_path: Path to model checkpoint
        """
        print(f"Loading model from: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path,
                              map_location=f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu')

        try:
            self.model.load_state_dict(checkpoint['model_state'])
        except:
            # Handle DistributedDataParallel models
            new_state_dict = OrderedDict()
            for key, val in checkpoint['model_state'].items():
                if key.startswith('module.'):
                    name = key[7:]  # Remove 'module.' prefix
                else:
                    name = key
                new_state_dict[name] = val
            self.model.load_state_dict(new_state_dict)

        # Store original parameters
        self.original_params = self._get_parameter_vector().clone()
        print(f"Model loaded successfully. Original parameters stored.")

        return checkpoint.get('epoch', 0), checkpoint.get('iters', 0)

    def generate_random_directions(self, normalize=True, filter_normalize=False):
        """
        Generate two random directions in parameter space for landscape exploration

        Args:
            normalize: If True, normalize directions to unit norm
            filter_normalize: If True, use filter-wise normalization (for convolutional layers)

        Returns:
            tuple: (direction1, direction2) as parameter vectors
        """
        print("Generating random directions for loss landscape exploration...")

        # Generate random Gaussian directions
        direction1 = torch.randn_like(self.original_params)
        direction2 = torch.randn_like(self.original_params)

        if filter_normalize:
            # Filter-wise normalization for better scaling across layers
            print("Applying filter-wise normalization...")
            direction1 = self._filter_normalize_direction(direction1)
            direction2 = self._filter_normalize_direction(direction2)

        if normalize:
            # Normalize to unit norm
            direction1 = direction1 / torch.norm(direction1)
            direction2 = direction2 / torch.norm(direction2)

            # Ensure orthogonality using Gram-Schmidt
            direction2 = direction2 - (torch.dot(direction1, direction2) * direction1)
            direction2 = direction2 / torch.norm(direction2)

        self.direction1 = direction1
        self.direction2 = direction2

        # Verify orthogonality
        dot_product = torch.dot(direction1, direction2).item()
        print(f"Direction norms: {torch.norm(direction1):.6f}, {torch.norm(direction2):.6f}")
        print(f"Dot product (should be ~0): {dot_product:.6f}")

        return direction1, direction2

    def _filter_normalize_direction(self, direction):
        """
        Apply filter-wise normalization to direction vector

        Args:
            direction: Parameter direction vector

        Returns:
            torch.Tensor: Filter-normalized direction
        """
        idx = 0
        for p in self.model.parameters():
            if p.requires_grad:
                param_size = p.numel()
                # Extract direction for this parameter
                p_direction = direction[idx:idx+param_size].view(p.shape)

                # Normalize by parameter norm
                p_norm = torch.norm(p.data)
                if p_norm > 1e-10:
                    p_direction = p_direction / torch.norm(p_direction) * p_norm

                # Put back into direction vector
                direction[idx:idx+param_size] = p_direction.reshape(-1)
                idx += param_size

        return direction

    def evaluate_loss_at_point(self, alpha, beta, data_loader, max_batches=None):
        """
        Evaluate loss at point: θ + α*d1 + β*d2

        Args:
            alpha: Coefficient for direction 1
            beta: Coefficient for direction 2
            data_loader: Data loader for loss computation
            max_batches: Maximum number of batches to process

        Returns:
            float: Loss value at the point
        """
        # Compute new parameter vector
        param_vector = self.original_params + alpha * self.direction1 + beta * self.direction2

        # Set model parameters
        self._set_parameter_vector(param_vector)
        self.model.eval()

        total_loss = 0.0
        total_err = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(data_loader):
                if max_batches is not None and batch_idx >= max_batches:
                    break

                if not hasattr(self.params, 'pack_data') or not self.params.pack_data:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)

                # Forward pass
                u = self.model(inputs)

                # Compute loss components
                loss_data = self.loss_func.data(inputs, u, targets)
                loss_pde = self.loss_func.pde(inputs, u, targets)
                loss_bc = self.loss_func.bc(inputs, u, targets)
                loss = loss_data + loss_bc + loss_pde

                # Compute error
                err = l2_err(u.detach(), targets.detach())

                total_loss += loss.item()
                total_err += err.item()
                num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        avg_err = total_err / max(num_batches, 1)

        return avg_loss, avg_err

    def compute_loss_surface(self, data_loader, x_range=(-0.5, 0.5), y_range=(-0.5, 0.5),
                            resolution=25, max_batches=None, normalize_directions=True,
                            filter_normalize=False, log_scale=True):
        """
        Compute 3D loss surface over a grid

        Args:
            data_loader: Data loader for loss computation
            x_range: (min, max) for direction 1
            y_range: (min, max) for direction 2
            resolution: Number of grid points in each direction
            max_batches: Maximum batches per evaluation
            normalize_directions: Normalize random directions
            filter_normalize: Use filter-wise normalization
            log_scale: Whether to log10-scale loss values for the z-axis

        Returns:
            dict: Results containing loss surface and metadata
        """
        if self.original_params is None:
            raise ValueError("Must load model checkpoint first using load_model()")

        # Generate random directions if not already done
        if self.direction1 is None or self.direction2 is None:
            self.generate_random_directions(normalize=normalize_directions,
                                           filter_normalize=filter_normalize)

        def normalize_range(rng):
            lower, upper = rng
            if lower > upper:
                lower, upper = upper, lower
            return lower, upper

        x_min, x_max = normalize_range(x_range)
        y_min, y_max = normalize_range(y_range)

        print(f"Computing {resolution}x{resolution} loss surface...")
        print(f"X range (direction 1): ({x_min:.2f}, {x_max:.2f})")
        print(f"Y range (direction 2): ({y_min:.2f}, {y_max:.2f})")

        # Create grid
        x_coords = np.linspace(x_min, x_max, resolution)
        y_coords = np.linspace(y_min, y_max, resolution)

        # Initialize loss and error surfaces
        loss_surface = np.zeros((resolution, resolution))
        error_surface = np.zeros((resolution, resolution))

        transform_loss = (lambda val: np.log10(val + 1e-12)) if log_scale else (lambda val: val)

        # Evaluate loss at center point (original model)
        print("Evaluating at center point (original model)...")
        center_loss_raw, center_err = self.evaluate_loss_at_point(0.0, 0.0, data_loader, max_batches)
        center_loss = transform_loss(center_loss_raw)
        print(f"Center loss: {center_loss_raw:.6f} (raw) -> {center_loss:.6f} ({'log10' if log_scale else 'linear'})")
        print(f"Center error: {center_err:.6f}")

        # Compute loss surface
        total_points = resolution * resolution
        evaluated_points = 0

        for i, alpha in enumerate(x_coords):
            for j, beta in enumerate(y_coords):
                loss_val, err_val = self.evaluate_loss_at_point(alpha, beta, data_loader, max_batches)
                loss_surface[j, i] = transform_loss(loss_val)  # Note: j, i for proper orientation
                error_surface[j, i] = err_val
                evaluated_points += 1

                if evaluated_points % max(1, total_points // 20) == 0:
                    progress = 100 * evaluated_points / total_points
                    print(f"Progress: {progress:.1f}% ({evaluated_points}/{total_points} points)")

        # Restore original parameters
        self._set_parameter_vector(self.original_params)

        # Compute statistics
        loss_stats = {
            'min': np.min(loss_surface),
            'max': np.max(loss_surface),
            'mean': np.mean(loss_surface),
            'std': np.std(loss_surface),
            'center': center_loss,
            'relative_center': (center_loss - np.min(loss_surface)) / max(np.max(loss_surface) - np.min(loss_surface), 1e-12)
        }

        results = {
            'loss_surface': loss_surface,
            'error_surface': error_surface,
            'x_coords': x_coords,
            'y_coords': y_coords,
            'x_range': (x_min, x_max),
            'y_range': (y_min, y_max),
            'resolution': resolution,
            'direction1_norm': torch.norm(self.direction1).item(),
            'direction2_norm': torch.norm(self.direction2).item(),
            'directions_dot_product': torch.dot(self.direction1, self.direction2).item(),
            'center_loss': center_loss,
            'center_loss_raw': center_loss_raw,
            'center_error': center_err,
            'log_scale': log_scale,
            'loss_stats': loss_stats,
            'num_params': self.n_params
        }

        print("\nLoss surface computed successfully!")
        print(f"Loss range: [{loss_stats['min']:.6f}, {loss_stats['max']:.6f}]")
        print(f"Center loss position: {loss_stats['relative_center']:.2%} of range")

        return results

    def _clamp_surface_results(self, results, max_radius=None):
        """
        Optionally restrict surface results to the allowed epsilon window.
        """
        if max_radius is None:
            return results

        x_coords = np.array(results['x_coords'])
        y_coords = np.array(results['y_coords'])

        x_mask = np.abs(x_coords) <= max_radius + 1e-12
        y_mask = np.abs(y_coords) <= max_radius + 1e-12

        if x_mask.all() and y_mask.all():
            return results

        clipped = dict(results)
        clipped['x_coords'] = x_coords[x_mask]
        clipped['y_coords'] = y_coords[y_mask]
        clipped['loss_surface'] = results['loss_surface'][np.ix_(y_mask, x_mask)]
        clipped['error_surface'] = results['error_surface'][np.ix_(y_mask, x_mask)]
        clipped['x_range'] = (-max_radius, max_radius)
        clipped['y_range'] = (-max_radius, max_radius)
        clipped['resolution'] = clipped['loss_surface'].shape[0]
        return clipped

    def compute_directional_loss_profile(self, data_loader, direction,
                                        range_vals=(-1.0, 1.0), num_points=50,
                                        max_batches=None):
        """
        Compute loss along a single direction

        Args:
            data_loader: Data loader for loss computation
            direction: Direction vector in parameter space
            range_vals: (min, max) range for direction coefficient
            num_points: Number of points to evaluate
            max_batches: Maximum batches per evaluation

        Returns:
            dict: Results containing loss profile
        """
        if self.original_params is None:
            raise ValueError("Must load model checkpoint first using load_model()")

        print(f"Computing loss profile along direction with {num_points} points...")

        # Normalize direction
        direction = direction / torch.norm(direction)

        # Create coefficient range
        coeffs = np.linspace(range_vals[0], range_vals[1], num_points)

        losses = []
        errors = []

        for i, coeff in enumerate(coeffs):
            # Compute parameter vector
            param_vector = self.original_params + coeff * direction

            # Set and evaluate
            self._set_parameter_vector(param_vector)
            self.model.eval()

            total_loss = 0.0
            total_err = 0.0
            num_batches = 0

            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(data_loader):
                    if max_batches is not None and batch_idx >= max_batches:
                        break

                    if not hasattr(self.params, 'pack_data') or not self.params.pack_data:
                        inputs, targets = inputs.to(self.device), targets.to(self.device)

                    u = self.model(inputs)

                    loss_data = self.loss_func.data(inputs, u, targets)
                    loss_pde = self.loss_func.pde(inputs, u, targets)
                    loss_bc = self.loss_func.bc(inputs, u, targets)
                    loss = loss_data + loss_bc + loss_pde

                    err = l2_err(u.detach(), targets.detach())

                    total_loss += loss.item()
                    total_err += err.item()
                    num_batches += 1

            avg_loss = total_loss / max(num_batches, 1)
            avg_err = total_err / max(num_batches, 1)

            losses.append(avg_loss)
            errors.append(avg_err)

            if i % max(1, num_points // 10) == 0:
                print(f"Point {i+1}/{num_points}: coeff={coeff:.3f}, loss={avg_loss:.6f}")

        # Restore original parameters
        self._set_parameter_vector(self.original_params)

        return {
            'coefficients': coeffs,
            'losses': np.array(losses),
            'errors': np.array(errors),
            'direction_norm': torch.norm(direction).item(),
            'range': range_vals,
            'num_points': num_points
        }

    def plot_loss_surface_3d(self, results, save_path=None, elev=30, azim=45):
        """
        Create 3D surface plot of loss landscape

        Args:
            results: Results from compute_loss_surface
            save_path: Optional path to save the plot
            elev: Elevation angle for 3D view
            azim: Azimuth angle for 3D view

        Returns:
            matplotlib.figure.Figure: The plot figure
        """
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
        except ImportError:
            print("Matplotlib not available for plotting")
            return None
        results = self._clamp_surface_results(results)
        log_scale = results.get('log_scale', False)

        fig = plt.figure(figsize=(14, 6))

        # 3D surface plot
        ax1 = fig.add_subplot(121, projection='3d')
        X, Y = np.meshgrid(results['x_coords'], results['y_coords'])
        surf = ax1.plot_surface(X, Y, results['loss_surface'], cmap='viridis',
                               alpha=0.9, edgecolor='none')

        # Mark center point
        ax1.scatter([0], [0], [results['center_loss']],
                   color='red', s=100, marker='o', label='Original model')

        ax1.set_xlabel(r"$\varepsilon_1$", fontsize=12)
        ax1.set_ylabel(r"$\varepsilon_2$", fontsize=12)
        ax1.set_zlabel('Loss (log10)' if log_scale else 'Loss')
        ax1.view_init(elev=elev, azim=azim)
        ax1.grid(False)
        max_eps = max(np.max(np.abs(results['x_coords'])), np.max(np.abs(results['y_coords'])))
        ax1.set_xlim(-max_eps, max_eps)
        ax1.set_ylim(-max_eps, max_eps)
        ax1.legend()

        # Add colorbar
        cbar1 = fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=5)
        cbar1.set_label('Loss (log10)' if log_scale else 'Loss')

        # 2D contour plot
        ax2 = fig.add_subplot(122)
        contour = ax2.contour(X, Y, results['loss_surface'], levels=20, cmap='viridis')
        ax2.contourf(X, Y, results['loss_surface'], levels=20, cmap='viridis', alpha=0.6)
        ax2.scatter([0], [0], color='red', s=100, marker='x',
                   linewidths=3, label='Original model')

        ax2.set_xlabel(r"$\varepsilon_1$", fontsize=12)
        ax2.set_ylabel(r"$\varepsilon_2$", fontsize=12)
        ax2.set_xlim(-max_eps, max_eps)
        ax2.set_ylim(-max_eps, max_eps)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Add colorbar
        cbar2 = fig.colorbar(contour, ax=ax2)
        cbar2.set_label('Loss (log10)' if log_scale else 'Loss')

        # Add statistics text
        stats_text = f"Loss range: [{results['loss_stats']['min']:.4f}, {results['loss_stats']['max']:.4f}]"
        if log_scale:
            stats_text += " (log10)"
        stats_text += f"\nCenter loss: {results.get('center_loss_raw', results['center_loss']):.4f}\n"
        stats_text += f"Resolution: {results['resolution']}×{results['resolution']}"

        fig.text(0.5, 0.02, stats_text, ha='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"3D loss landscape plot saved to {save_path}")

        return fig

    def plot_loss_surface_2d_heatmap(self, results, save_path=None):
        """
        Create 2D heatmap of loss landscape

        Args:
            results: Results from compute_loss_surface
            save_path: Optional path to save the plot

        Returns:
            matplotlib.figure.Figure: The plot figure
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Matplotlib not available for plotting")
            return None
        results = self._clamp_surface_results(results)
        log_scale = results.get('log_scale', False)

        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        # Create heatmap
        X, Y = np.meshgrid(results['x_coords'], results['y_coords'])
        im = ax.pcolormesh(X, Y, results['loss_surface'], cmap='viridis', shading='auto')

        # Mark center point
        ax.scatter([0], [0], color='red', s=200, marker='x',
                  linewidths=4, label='Original model', zorder=5)

        # Add contour lines
        contour = ax.contour(X, Y, results['loss_surface'], levels=15,
                           colors='white', alpha=0.4, linewidths=0.5)
        ax.clabel(contour, inline=True, fontsize=8, fmt='%.3f')

        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Loss (log10)' if log_scale else 'Loss', rotation=270, labelpad=20)

        ax.set_xlabel(r"$\varepsilon_1$", fontsize=12)
        ax.set_ylabel(r"$\varepsilon_2$", fontsize=12)
        ax.set_title('Loss Landscape Heatmap' + (' (log10 scale)' if log_scale else ''), fontsize=14)
        max_eps = max(np.max(np.abs(results['x_coords'])), np.max(np.abs(results['y_coords'])))
        ax.set_xlim(-max_eps, max_eps)
        ax.set_ylim(-max_eps, max_eps)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, color='white', linewidth=0.5)

        # Add statistics
        stats_text = (
            f"{'Log10' if log_scale else 'Linear'} range: {results['loss_stats']['min']:.4f}–{results['loss_stats']['max']:.4f} | "
            f"Center loss (raw): {results.get('center_loss_raw', results['center_loss']):.4f}"
        )
        ax.text(0.5, -0.1, stats_text, transform=ax.transAxes,
               ha='center', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"2D loss landscape heatmap saved to {save_path}")

        return fig

    @classmethod
    def from_config(cls, params, checkpoint_path=None):
        """
        Create LossLandscape3D from config parameters

        Args:
            params: Configuration parameters (YParams object)
            checkpoint_path: Optional path to model checkpoint

        Returns:
            LossLandscape3D: Initialized calculator
        """
        calc = cls(params)

        if checkpoint_path is not None:
            calc.load_model(checkpoint_path)

        return calc
