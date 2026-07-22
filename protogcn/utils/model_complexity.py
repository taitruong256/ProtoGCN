"""Utilities for logging model architecture, parameters, and FLOPs."""

import torch
import torch.nn as nn

__all__ = ['log_model_parameters', 'log_model_flops']


def _module_param_counts(module):
    """Return ``(total, trainable, own, children)`` for a module tree."""
    own_params = list(module.parameters(recurse=False))
    own = sum(param.numel() for param in own_params)
    own_trainable = sum(
        param.numel() for param in own_params if param.requires_grad)

    children = 0
    children_trainable = 0
    for child in module.children():
        child_total, child_trainable, _, _ = _module_param_counts(child)
        children += child_total
        children_trainable += child_trainable

    return own + children, own_trainable + children_trainable, own, children


def _log_module_tree(logger, module, name='model', indent=0):
    total, trainable, own, children = _module_param_counts(module)
    if total == 0:
        return

    prefix = '  ' * indent
    kind = 'layer' if not any(_module_param_counts(child)[0] for child in module.children()) else 'module'
    logger.info(
        '%s%-56s %-30s %-8s total=%15d own=%15d children=%15d trainable=%15d',
        prefix,
        name,
        module.__class__.__name__,
        kind,
        total,
        own,
        children,
        trainable,
    )

    # Children are printed only after their parent module, preserving the
    # stage -> submodule -> layer order in the log.
    for child_name, child in module.named_children():
        if _module_param_counts(child)[0] > 0:
            _log_module_tree(logger, child, child_name, indent + 1)


def log_model_parameters(logger, model):
    """Log model architecture and parameters as a hierarchical module tree.

    For every module, ``total = own + children``. This makes each stage's
    reported total directly checkable against the parameters of its layers.
    """
    logger.info('Model architecture:\n%s', model)
    logger.info('Model parameters by module and layer:')
    logger.info(
        '%-58s %-30s %-8s %15s %15s %15s %15s',
        'Module/layer', 'Type', 'Kind', 'Total', 'Own', 'Children', 'Trainable')
    logger.info('%s', '-' * 145)
    _log_module_tree(logger, model)
    total_params, total_trainable, _, _ = _module_param_counts(model)
    logger.info('%s', '-' * 145)
    logger.info(
        'Total parameters: %d | Trainable parameters: %d | Non-trainable parameters: %d',
        total_params,
        total_trainable,
        total_params - total_trainable,
    )


class _FlopsForward(nn.Module):
    """Inference-only wrapper used by fvcore to trace the recognizer."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, keypoint):
        features, _ = self.model.extract_feat(keypoint)
        return self.model.cls_head(features)


def _pipeline_value(cfg, pipeline_name, key, default):
    for transform in cfg.get(pipeline_name, []):
        if transform.get('type') == key:
            return transform.get('clip_len', default)
    return default


def _num_person(cfg):
    for pipeline_name in ('train_pipeline', 'val_pipeline'):
        for transform in cfg.get(pipeline_name, []):
            if transform.get('type') == 'FormatGCNInput':
                return transform.get('num_person', 1)
    return 1


def log_model_flops(logger, model, cfg, batch_size=16):
    """Log FLOPs for a batch and the average FLOPs per sample."""
    if batch_size <= 0:
        raise ValueError('batch_size must be greater than zero')

    try:
        from fvcore.nn import FlopCountAnalysis
    except ImportError:
        logger.warning('FLOPs were not logged because fvcore is not installed.')
        return

    backbone_cfg = cfg.model.backbone
    channels = backbone_cfg.get('in_channels', 3)
    frames = _pipeline_value(cfg, 'train_pipeline', 'UniformSampleDecode', 100)
    joints = model.backbone.graph.num_node
    persons = _num_person(cfg)
    device = next(model.parameters()).device
    dummy_input = torch.randn(
        batch_size, persons, frames, joints, channels, device=device)

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            flops = FlopCountAnalysis(_FlopsForward(model), dummy_input).total()
    finally:
        model.train(was_training)

    logger.info(
        'FLOPs: batch_size=%d, input_shape=%s, total=%d FLOPs (%.9f GFLOPs), average=%d/%d FLOPs/sample (%.9f GFLOPs/sample)',
        batch_size,
        tuple(dummy_input.shape),
        int(flops),
        flops / 1e9,
        int(flops),
        batch_size,
        flops / batch_size / 1e9,
    )
