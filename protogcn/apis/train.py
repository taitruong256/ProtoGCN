import numpy as np
import os
import os.path as osp
import time
import torch
import torch.distributed as dist
from mmcv.engine import multi_gpu_test
from mmcv.parallel import MMDistributedDataParallel
from mmcv.runner import DistSamplerSeedHook, EpochBasedRunner, Hook, OptimizerHook, build_optimizer, get_dist_info

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


class DistPrefixEvalHook(Hook):
    """Run distributed evaluation after each epoch and log with a prefix."""

    def __init__(self, dataset, dataloader, eval_cfg=None, prefix='train', tmpdir=None):
        self.dataset = dataset
        self.dataloader = dataloader
        self.eval_cfg = dict(eval_cfg or {})
        self.prefix = prefix
        self.tmpdir = tmpdir

    def after_train_epoch(self, runner):
        outputs = multi_gpu_test(runner.model, self.dataloader, self.tmpdir)
        rank, _ = get_dist_info()
        if rank != 0:
            return

        eval_cfg = dict(self.eval_cfg)
        for key in [
                'interval', 'tmpdir', 'start',
                'save_best', 'rule', 'by_epoch', 'broadcast_bn_buffers'
        ]:
            eval_cfg.pop(key, None)

        eval_cfg.setdefault('split_name', self.prefix)
        eval_res = self.dataset.evaluate(outputs, **eval_cfg)
        runner.logger.info(f'{self.prefix} results at epoch {runner.epoch + 1}')
        for metric_name, val in eval_res.items():
            runner.logger.info(f'{self.prefix}_{metric_name}: {val:.04f}')


def _build_prefixed_eval_hook(cfg, split_cfg, prefix, default_eval_cfg_key, dataloader_cfg_key):
    if split_cfg is None:
        return None, None

    split_dataset = build_dataset(split_cfg, dict(test_mode=True))
    dataloader_setting = dict(
        videos_per_gpu=cfg.data.get('videos_per_gpu', 1),
        workers_per_gpu=cfg.data.get('workers_per_gpu', 1),
        persistent_workers=cfg.data.get('persistent_workers', False),
        shuffle=False)
    dataloader_setting = dict(dataloader_setting,
                              **cfg.data.get(dataloader_cfg_key, {}))
    split_dataloader = build_dataloader(split_dataset, **dataloader_setting)
    split_eval_cfg = cfg.get(default_eval_cfg_key, cfg.get('evaluation', {}))
    tmpdir = cfg.get('evaluation', {}).get('tmpdir', osp.join(cfg.work_dir, 'tmp'))
    hook = DistPrefixEvalHook(
        split_dataset,
        split_dataloader,
        split_eval_cfg,
        prefix=prefix,
        tmpdir=tmpdir)
    return split_dataset, hook


def _apply_class_weight_if_needed(model, train_dataset, cfg, logger):
    if not cfg.get('use_class_weight', False):
        return
    class_weight = getattr(train_dataset, 'class_weight', None)
    if class_weight is None:
        logger.info('Class weight is enabled, but train dataset did not provide class_weight.')
        return

    try:
        target = model.module if hasattr(model, 'module') else model
        loss_obj = target.cls_head.loss_cls
        if hasattr(loss_obj, 'class_weight'):
            loss_obj.class_weight = torch.tensor(class_weight, dtype=torch.float32)
            logger.info(f'Applied class_weight to loss: {class_weight}')
        else:
            logger.info('Class weight is enabled, but loss object has no class_weight field.')
    except Exception as exc:
        logger.warning(f'Failed to apply class_weight: {exc}')


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

    # put model on gpus
    find_unused_parameters = cfg.get('find_unused_parameters', True)
    # Sets the `find_unused_parameters` parameter in
    # torch.nn.parallel.DistributedDataParallel
    
    model = MMDistributedDataParallel(
        model.cuda(),
        device_ids=[torch.cuda.current_device()],
        broadcast_buffers=False,
        find_unused_parameters=find_unused_parameters)

    _apply_class_weight_if_needed(model, dataset[0], cfg, logger)

    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)

    Runner = EpochBasedRunner
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
    train_eval_hook = None
    test_eval_hook = None

    if cfg.get('use_val', True) and validate and cfg.data.get('val', None) is not None:
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

    if cfg.data.get('train_eval', None) is not None:
        _, train_eval_hook = _build_prefixed_eval_hook(
            cfg,
            cfg.data.train_eval,
            prefix='train',
            default_eval_cfg_key='train_evaluation',
            dataloader_cfg_key='train_eval_dataloader')
        runner.register_hook(train_eval_hook)

    if cfg.data.get('test_eval', None) is not None:
        _, test_eval_hook = _build_prefixed_eval_hook(
            cfg,
            cfg.data.test_eval,
            prefix='test',
            default_eval_cfg_key='test_evaluation',
            dataloader_cfg_key='test_eval_dataloader')
        runner.register_hook(test_eval_hook)

    if cfg.get('resume_from', None):
        runner.resume(cfg.resume_from)
    elif cfg.get('load_from', None):
        cfg.load_from = cache_checkpoint(cfg.load_from)
        runner.load_checkpoint(cfg.load_from)

    runner.run(data_loaders, cfg.workflow, cfg.total_epochs)

    dist.barrier()
    time.sleep(2)

    if test['test_last'] or test['test_best']:
        best_ckpt_path = None
        if test['test_best']:
            if eval_hook is None:
                logger.info(
                    'Warning: test_best requested without validation hook. '
                    'Falling back to last checkpoint evaluation.')
                test['test_last'] = True
                test['test_best'] = False
            else:
                ckpt_paths = [x for x in os.listdir(cfg.work_dir) if 'best' in x]
                ckpt_paths = [x for x in ckpt_paths if x.endswith('.pth')]
                if len(ckpt_paths) == 0:
                    logger.info('Warning: test_best set, but no ckpt found')
                    test['test_best'] = False
                    if not test['test_last']:
                        return
                elif len(ckpt_paths) > 1:
                    epoch_ids = [
                        int(x.split('epoch_')[-1][:-4]) for x in ckpt_paths
                    ]
                    best_ckpt_path = ckpt_paths[np.argmax(epoch_ids)]
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
                eval_cfg.setdefault('split_name', name)

                eval_res = test_dataset.evaluate(outputs, **eval_cfg)
                logger.info(f'Testing results of the {name} checkpoint')
                for metric_name, val in eval_res.items():
                    logger.info(f'{metric_name}: {val:.04f}')
