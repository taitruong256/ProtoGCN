import numpy as np
import os
import os.path as osp
import re
import time
import torch
import torch.distributed as dist
from fvcore.nn import FlopCountAnalysis, parameter_count
from mmcv.engine import multi_gpu_test
from mmcv.parallel import MMDistributedDataParallel
from mmcv.runner import (DistSamplerSeedHook, EpochBasedRunner,
                         IterBasedRunner, OptimizerHook, build_optimizer,
                         get_dist_info)

from ..core import DistEvalHook
from ..datasets import build_dataloader, build_dataset
from ..utils import cache_checkpoint, get_root_logger


def init_random_seed(seed=None, device='cuda'):
    """Initialize random seed.

    If the seed is not set, the seed will be automatically randomized,
    and then broadcast to all processes to prevent some potential bugs.
    Args:
        seed (int, Optional): The seed. Default to None.
        device (str): The device where the seed will be put on.
            Default to 'cuda'.
    Returns:
        int: Seed to be used.
    """
    if seed is not None:
        return seed

    rank, world_size = get_dist_info()
    seed = np.random.randint(2**31)

    if world_size == 1:
        return seed

    if rank == 0:
        random_num = torch.tensor(seed, dtype=torch.int32, device=device)
    else:
        random_num = torch.tensor(0, dtype=torch.int32, device=device)

    dist.broadcast(random_num, src=0)
    return random_num.item()


class _ComplexityWrapper(torch.nn.Module):
    """Wrap the recognizer so FLOPs are measured on tensor inputs only."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, keypoint):
        if keypoint.dim() == 6 and keypoint.size(1) == 1:
            keypoint = keypoint[:, 0]
        x, _ = self.model.extract_feat(keypoint)
        if self.model.with_cls_head:
            x = self.model.cls_head(x)
        return x


def _count_params(module):
    return sum(p.numel() for p in module.parameters())


def _log_module_tree(module, logger, prefix='', depth=0):
    """Recursively log a readable module tree with parameter counts."""
    children = list(module.named_children())
    indent = '  ' * depth
    cls_name = module.__class__.__name__
    params = _count_params(module)

    if prefix:
        logger.info('%s%s: %s | params=%d', indent, prefix, cls_name, params)
    else:
        logger.info('%s%s | params=%d', indent, cls_name, params)

    for name, child in children:
        _log_module_tree(child, logger, name, depth + 1)


def _log_branch_params(model, logger):
    """Log parameter totals for spatial and temporal branches separately."""
    backbone = getattr(model, 'backbone', None)
    if backbone is None or not hasattr(backbone, 'gcn'):
        logger.info('Branch parameter breakdown skipped: backbone.gcn not found.')
        return

    spatial_params = 0
    temporal_params = 0
    residual_params = 0

    for idx, block in enumerate(backbone.gcn):
        if hasattr(block, 'gcn'):
            spatial_params += _count_params(block.gcn)
        if hasattr(block, 'tcn'):
            temporal_params += _count_params(block.tcn)
        if hasattr(block, 'residual') and isinstance(block.residual, torch.nn.Module):
            residual_params += _count_params(block.residual)

    other_params = _count_params(backbone) - spatial_params - temporal_params - residual_params

    logger.info('Branch parameter breakdown')
    logger.info('  spatial_gcn_params: %d', spatial_params)
    logger.info('  temporal_tcn_params: %d', temporal_params)
    logger.info('  residual_params: %d', residual_params)
    logger.info('  other_backbone_params: %d', other_params)


def _log_model_complexity(model, data_loader, logger, device):
    """Log exact parameter count and FLOPs for one real batch."""
    batch = next(iter(data_loader))
    if 'keypoint' not in batch:
        raise KeyError('Batch does not contain `keypoint`, cannot compute FLOPs.')

    keypoint = batch['keypoint']
    if torch.is_tensor(keypoint):
        keypoint = keypoint.to(device)
    else:
        raise TypeError(f'Unsupported keypoint type for FLOPs analysis: {type(keypoint)}')

    if keypoint.size(0) < 1:
        raise ValueError('Keypoint batch is empty, cannot compute FLOPs.')

    sample_keypoint = keypoint[:1]
    complexity_model = _ComplexityWrapper(model).to(device).eval()
    params = parameter_count(complexity_model)['']
    with torch.no_grad():
        flops = FlopCountAnalysis(complexity_model, sample_keypoint).total()

    logger.info('Model complexity summary')
    logger.info('Model layer tree')
    _log_module_tree(model, logger)
    logger.info('  parameters: %d', params)
    logger.info('  flops_per_sample: %d', flops)
    _log_branch_params(model, logger)
    return params, flops


def train_model(model,
                dataset,
                cfg,
                validate=False,
                test=dict(test_best=False, test_last=False),
                timestamp=None,
                meta=None):
    """Train model entry function.

    Args:
        model (nn.Module): The model to be trained.
        dataset (:obj:`Dataset`): Train dataset.
        cfg (dict): The config dict for training.
        validate (bool): Whether to do evaluation. Default: False.
        test (dict): The testing option, with two keys: test_last & test_best.
            The value is True or False, indicating whether to test the
            corresponding checkpoint.
            Default: dict(test_best=False, test_last=False).
        timestamp (str | None): Local time for runner. Default: None.
        meta (dict | None): Meta dict to record some important information.
            Default: None
    """
    logger = get_root_logger(log_level=cfg.get('log_level', 'INFO'))

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]

    dataloader_setting = dict(
        videos_per_gpu=cfg.data.get('videos_per_gpu', 1),
        workers_per_gpu=cfg.data.get('workers_per_gpu', 1),
        persistent_workers=cfg.data.get('persistent_workers', False),
        seed=cfg.seed)
    dataloader_setting = dict(dataloader_setting,
                              **cfg.data.get('train_dataloader', {}))

    data_loaders = [
        build_dataloader(ds, **dataloader_setting) for ds in dataset
    ]

    rank, world_size = get_dist_info()
    if rank == 0:
        device = torch.device('cuda', torch.cuda.current_device()) if torch.cuda.is_available() else torch.device('cpu')
        _log_model_complexity(model, data_loaders[0], logger, device)
    if dist.is_available() and dist.is_initialized() and world_size > 1:
        dist.barrier()

    # put model on gpus
    find_unused_parameters = cfg.get('find_unused_parameters', True)
    # Sets the `find_unused_parameters` parameter in
    # torch.nn.parallel.DistributedDataParallel
    
    model = MMDistributedDataParallel(
        model.cuda(),
        device_ids=[torch.cuda.current_device()],
        broadcast_buffers=False,
        find_unused_parameters=find_unused_parameters)

    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)

    use_iter_runner = cfg.get('runner_type', None) == 'IterBasedRunner' or 'total_iters' in cfg
    Runner = IterBasedRunner if use_iter_runner else EpochBasedRunner
    runner = Runner(
        model,
        optimizer=optimizer,
        work_dir=cfg.work_dir,
        logger=logger,
        meta=meta)
    # an ugly workaround to make .log and .log.json filenames the same
    runner.timestamp = timestamp

    if 'type' not in cfg.optimizer_config:
        optimizer_config = OptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config

    # register hooks
    runner.register_training_hooks(cfg.lr_config, optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config,
                                   cfg.get('momentum_config', None))
    runner.register_hook(DistSamplerSeedHook())

    eval_hook = None
    if validate:
        eval_cfg = cfg.get('evaluation', {})
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        dataloader_setting = dict(
            videos_per_gpu=cfg.data.get('videos_per_gpu', 1),
            workers_per_gpu=cfg.data.get('workers_per_gpu', 1),
            persistent_workers=cfg.data.get('persistent_workers', False),
            shuffle=False)
        dataloader_setting = dict(dataloader_setting,
                                  **cfg.data.get('val_dataloader', {}))
        val_dataloader = build_dataloader(val_dataset, **dataloader_setting)
        eval_hook = DistEvalHook(val_dataloader, **eval_cfg)
        runner.register_hook(eval_hook)

    if cfg.get('resume_from', None):
        runner.resume(cfg.resume_from)
    elif cfg.get('load_from', None):
        cfg.load_from = cache_checkpoint(cfg.load_from)
        runner.load_checkpoint(cfg.load_from)

    max_progress = cfg.total_iters if use_iter_runner else cfg.total_epochs
    runner.run(data_loaders, cfg.workflow, max_progress)

    dist.barrier()
    time.sleep(2)

    if test['test_last'] or test['test_best']:
        best_ckpt_path = None
        if test['test_best']:
            assert eval_hook is not None
            best_ckpt_path = None
            ckpt_paths = [x for x in os.listdir(cfg.work_dir) if 'best' in x]
            ckpt_paths = [x for x in ckpt_paths if x.endswith('.pth')]
            if len(ckpt_paths) == 0:
                logger.info('Warning: test_best set, but no ckpt found')
                test['test_best'] = False
                if not test['test_last']:
                    return
            elif len(ckpt_paths) > 1:
                ckpt_ids = []
                for path in ckpt_paths:
                    match = re.search(r'(?:epoch|iter)_(\d+)', path)
                    ckpt_ids.append(int(match.group(1)) if match else -1)
                best_ckpt_path = ckpt_paths[np.argmax(ckpt_ids)]
            else:
                best_ckpt_path = ckpt_paths[0]
            if best_ckpt_path:
                best_ckpt_path = osp.join(cfg.work_dir, best_ckpt_path)

        test_dataset = build_dataset(cfg.data.test, dict(test_mode=True))
        tmpdir = cfg.get('evaluation', {}).get('tmpdir', osp.join(cfg.work_dir, 'tmp'))
        dataloader_setting = dict(
            videos_per_gpu=cfg.data.get('videos_per_gpu', 1),
            workers_per_gpu=cfg.data.get('workers_per_gpu', 1),
            persistent_workers=cfg.data.get('persistent_workers', False),
            shuffle=False)
        dataloader_setting = dict(dataloader_setting,
                                  **cfg.data.get('test_dataloader', {}))

        test_dataloader = build_dataloader(test_dataset, **dataloader_setting)

        names, ckpts = [], []

        if test['test_last']:
            names.append('last')
            ckpts.append(None)
        if test['test_best']:
            names.append('best')
            ckpts.append(best_ckpt_path)

        for name, ckpt in zip(names, ckpts):
            if ckpt is not None:
                runner.load_checkpoint(ckpt)

            outputs = multi_gpu_test(runner.model, test_dataloader, tmpdir)
            rank, _ = get_dist_info()
            if rank == 0:
                out = osp.join(cfg.work_dir, f'{name}_pred.pkl')
                test_dataset.dump_results(outputs, out)

                eval_cfg = cfg.get('evaluation', {})
                for key in [
                        'interval', 'tmpdir', 'start',
                        'save_best', 'rule', 'by_epoch', 'broadcast_bn_buffers'
                ]:
                    eval_cfg.pop(key, None)

                eval_res = test_dataset.evaluate(outputs, **eval_cfg)
                logger.info(f'Testing results of the {name} checkpoint')
                for metric_name, val in eval_res.items():
                    logger.info(f'{metric_name}: {val:.04f}')
