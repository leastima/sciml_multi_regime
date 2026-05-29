import os, sys, time
import argparse
import torch
import wandb
import matplotlib.pyplot as plt
import logging
import torch.distributed as dist
from utils import logging_utils
logging_utils.config_logger()
from utils.YParams import YParams
from utils.trainer import Trainer

if __name__ == '__main__':
    # parsers
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_config", default='./config/operators.yaml', type=str)
    parser.add_argument("--config", default='default', type=str)
    parser.add_argument("--root_dir", default='./', type=str, help='root dir to store results')
    parser.add_argument("--run_num", default='0', type=str, help='sub run config')
    parser.add_argument("--sweep_id", default=None, type=str, help='sweep config from ./configs/sweeps.yaml')
    # other parameters
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--subsample", default=32, type=int)
    parser.add_argument("--seed", default=2024, type=int)
    parser.add_argument("--max_epochs", default=500, type=int)
    args = parser.parse_args()
    
    # init params
    params = YParams(os.path.abspath(args.yaml_config), args.config)
    params['lr'] = args.lr
    params['batch_size'] = args.batch_size
    params['subsample'] = args.subsample
    params['seed'] = args.seed
    params['max_epochs'] = args.max_epochs
    trainer = Trainer(params, args)

    # check if the file is exist
    logfile_dir = os.path.join(*[args.root_dir, f'expts_eps{args.max_epochs}', args.config, args.run_num, f'bsz{args.batch_size}_lr{args.lr}_subsample{args.subsample}/seed{args.seed}/logs.txt'])
    if os.path.exists(logfile_dir):
        with open(logfile_dir, 'r', encoding='utf-8') as f:
            for line in f:
                if "epoch,499" in line:
                    logging.info('The model has been trained, please check the checkpoints folder.')
                    exit(0) 
        
    
    if args.sweep_id and trainer.world_rank==0:
        logging.disable(logging.CRITICAL)
        wandb.agent(args.sweep_id, function=trainer.launch, count=1, entity=trainer.params.entity, project=trainer.params.project) 
    else:
        trainer.launch()

    if dist.is_initialized():
        dist.barrier()

    logging.info('DONE')
