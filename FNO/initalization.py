import os
import argparse
import logging
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap as ruamelDict

from utils import logging_utils
logging_utils.config_logger()
from utils.YParams import YParams
from utils.trainer import Trainer, set_seed
from utils.optimizer_utils import set_optimizer, set_scheduler

import models.fno


def dump_hparams(params, exp_dir, world_rank):
    if world_rank != 0:
        return
    hparams = ruamelDict()
    yaml = YAML()
    for key, value in params.params.items():
        hparams[str(key)] = str(value)
    with open(os.path.join(exp_dir, 'hyperparams.yaml'), 'w', encoding='utf-8') as hpfile:
        yaml.dump(hparams, hpfile)


def build_model(params, device, local_rank):
    if params.model == 'fno':
        model = models.fno.fno(params).to(device)
    else:
        raise ValueError("Error, model arch invalid.")
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank], output_device=[local_rank])
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_config", default='./config/operators.yaml', type=str)
    parser.add_argument("--config", default='default', type=str)
    parser.add_argument("--root_dir", default='./', type=str, help='root dir to store results')
    parser.add_argument("--run_num", default='0', type=str, help='sub run config')
    parser.add_argument("--sweep_id", default=None, type=str, help='sweep config from ./configs/sweeps.yaml')
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--subsample", default=32, type=int)
    parser.add_argument("--seed", default=2024, type=int)
    parser.add_argument("--max_epochs", default=500, type=int)
    args = parser.parse_args()

    params = YParams(os.path.abspath(args.yaml_config), args.config)
    params['lr'] = args.lr
    params['batch_size'] = args.batch_size
    params['subsample'] = args.subsample
    params['seed'] = args.seed
    params['max_epochs'] = args.max_epochs

    if not hasattr(params, 'valid_batch_size'):
        params['valid_batch_size'] = params.batch_size

    trainer = Trainer(params, args)
    trainer.modify_bs_for_subsampling()

    exp_dir = os.path.join(
        *[
            args.root_dir,
            f'expts_eps{params.max_epochs}',
            args.config,
            args.run_num,
            f'bsz{params.batch_size}_lr{params.lr}_subsample{params.subsample}/seed{params.seed}',
        ]
    )
    trainer.init_exp_dir(exp_dir)

    init_ckpt_path = os.path.join(exp_dir, 'checkpoints/ckpt_init.tar')
    if os.path.exists(init_ckpt_path):
        logging.info('Initialization checkpoint exists, skipping: %s', init_ckpt_path)
        if dist.is_initialized():
            dist.barrier()
        raise SystemExit(0)

    set_seed(params, trainer.world_size)

    params['global_batch_size'] = params.batch_size
    params['local_batch_size'] = int(params.batch_size // trainer.world_size)
    params['global_valid_batch_size'] = params.valid_batch_size
    params['local_valid_batch_size'] = int(params.valid_batch_size // trainer.world_size)

    dump_hparams(params, exp_dir, trainer.world_rank)

    model = build_model(params, trainer.device, trainer.local_rank)
    optimizer = set_optimizer(params, model)
    scheduler = set_scheduler(params, optimizer)

    if trainer.world_rank == 0:
        torch.save(
            {
                'iters': 0,
                'epoch': 0,
                'model_state': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': (scheduler.state_dict() if scheduler is not None else None),
            },
            init_ckpt_path,
        )
        logging.info('Saved initialization checkpoint: %s', init_ckpt_path)

    if dist.is_initialized():
        dist.barrier()

    logging.info('DONE')
