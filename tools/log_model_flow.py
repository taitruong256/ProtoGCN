"""Log model architecture and one forward pass flow to a file.

This script is meant for understanding how tensors move through the
RecognizerGCN -> ProtoGCN -> GCN blocks -> SimpleHead path.
"""

import argparse
import logging
from pathlib import Path

import torch
from mmcv import Config
from mmcv.runner import load_checkpoint

from protogcn.datasets import build_dataset, build_dataloader
from protogcn.models import build_model
from protogcn.utils import cache_checkpoint, get_root_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Log ProtoGCN model flow")
    parser.add_argument("config", help="Config file path")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path or URL. Defaults to latest.pth in work_dir.")
    parser.add_argument("--split", default="val", choices=["train", "val"], help="Dataset split used for the dry run.")
    parser.add_argument("--log-file", default=None, help="Path to the output log file.")
    parser.add_argument("--device", default="cpu", help="Device for the dry run, e.g. cpu or cuda:0.")
    parser.add_argument("--sample-index", type=int, default=0, help="Batch index to inspect from the dataloader.")
    return parser.parse_args()


def _to_device(data, device):
    if torch.is_tensor(data):
        return data.to(device)
    if isinstance(data, dict):
        return {k: _to_device(v, device) for k, v in data.items()}
    if isinstance(data, list):
        return [_to_device(v, device) for v in data]
    if isinstance(data, tuple):
        return tuple(_to_device(v, device) for v in data)
    return data


def _shape_of(obj):
    if torch.is_tensor(obj):
        return tuple(obj.shape)
    if isinstance(obj, (list, tuple)):
        return [_shape_of(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _shape_of(v) for k, v in obj.items()}
    return type(obj).__name__


def _count_params(module):
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def _module_title(name, module):
    total, trainable = _count_params(module)
    return f"{name}: {module.__class__.__name__} | params={total} | trainable={trainable}"


def _log_module_tree(logger, module, name="model", indent=0):
    prefix = "  " * indent
    total, trainable = _count_params(module)
    logger.info(
        "%s%s: %s | params=%d | trainable=%d",
        prefix,
        name,
        module.__class__.__name__,
        total,
        trainable,
    )
    for child_name, child in module.named_children():
        _log_module_tree(logger, child, child_name, indent + 1)


def _module_tree_lines(module, name="model", indent=0, max_depth=None):
    prefix = "  " * indent
    total, trainable = _count_params(module)
    lines = [f"{prefix}- {name}: {module.__class__.__name__} | params={total} | trainable={trainable}"]
    if max_depth is not None and indent >= max_depth:
        return lines
    for child_name, child in module.named_children():
        lines.extend(_module_tree_lines(child, child_name, indent + 1, max_depth=max_depth))
    return lines


def _main_modules_section(model):
    lines = []
    for child_name, child in model.named_children():
        lines.append(f"- {_module_title(child_name, child)}")
    return lines


def _overall_architecture_section(model):
    lines = []
    lines.append("Input keypoint")
    lines.append("  -> RecognizerGCN")
    if hasattr(model, "backbone"):
        lines.append("     -> backbone (ProtoGCN)")
        lines.append("        -> data normalization / reshape")
        lines.append("        -> stacked GCN_Block stages")
        lines.append("           -> unit_gcn")
        lines.append("           -> mstcn + residual")
        lines.append("        -> graph reconstruction via PRN")
    if hasattr(model, "cls_head"):
        lines.append("     -> cls_head (SimpleHead)")
        lines.append("        -> pooling if input is 5D")
        lines.append("        -> fc_cls")
    lines.append("Output")
    lines.append("  -> train: losses")
    lines.append("  -> test: class scores / features")
    return lines


def _module_details_section(model):
    lines = []
    for child_name, child in model.named_children():
        lines.append(f"## {child_name}")
        lines.extend(_module_tree_lines(child, name=child_name, indent=0))
        lines.append("")
    return lines


def _full_model_details_section(model):
    return _module_tree_lines(model, name="model", indent=0)


def write_architecture_report(model, path):
    lines = []
    lines.append("# Model Architecture")
    lines.append("")
    lines.append("## 1. Main Modules")
    lines.extend(_main_modules_section(model))
    lines.append("")
    lines.append("## 2. Overall Architecture")
    lines.extend(f"- {line}" if not line.startswith("-") else line for line in _overall_architecture_section(model))
    lines.append("")
    lines.append("## 3. Module Details")
    lines.extend(_module_details_section(model))
    lines.append("## 4. Full Model Detail")
    lines.extend(_full_model_details_section(model))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _select_dataset_cfg(cfg, split):
    if split not in cfg.data:
        raise KeyError(f"Config does not contain data.{split}")
    return cfg.data.get(split)


def _select_dataloader_cfg(cfg, split):
    key = f"{split}_dataloader"
    if key in cfg.data:
        return cfg.data.get(key).copy()
    return {}


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    work_dir = Path(cfg.get("work_dir", Path(args.config).resolve().parent))
    log_file = Path(args.log_file) if args.log_file else work_dir / "model_flow.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = get_root_logger(log_file=str(log_file), log_level=logging.DEBUG)
    logger.info("Loading config from %s", args.config)
    logger.info("Writing detailed model flow log to %s", log_file)

    if cfg.model.get("backbone") is not None:
        cfg.model.backbone.pretrained = None
    checkpoint = args.checkpoint
    if checkpoint is None:
        latest = work_dir / "latest.pth"
        if latest.exists():
            checkpoint = str(latest)
    if checkpoint is not None:
        checkpoint = cache_checkpoint(checkpoint)

    model = build_model(cfg.model)
    if checkpoint is not None:
        logger.info("Loading checkpoint: %s", checkpoint)
        load_checkpoint(model, checkpoint, map_location="cpu")
    else:
        logger.info("No checkpoint found. The model will be logged with initialized weights.")

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA is not available, falling back to CPU.")
        args.device = "cpu"
    model = model.to(args.device)
    model.eval()

    arch_file = log_file.with_suffix(".architecture.md")
    write_architecture_report(model, arch_file)
    logger.info("Architecture report saved to %s", arch_file)

    logger.info("=== Model Summary ===")
    _log_module_tree(logger, model)

    dataset_cfg = _select_dataset_cfg(cfg, args.split)
    dataloader_cfg = _select_dataloader_cfg(cfg, args.split)
    dataloader_cfg.update(dict(videos_per_gpu=1, workers_per_gpu=0, shuffle=False, drop_last=False))

    logger.info("Building %s dataset and dataloader", args.split)
    dataset = build_dataset(dataset_cfg, dict(test_mode=(args.split != "train")))
    data_loader = build_dataloader(dataset, **dataloader_cfg)

    batch = None
    for i, data in enumerate(data_loader):
        if i == args.sample_index:
            batch = data
            break
    if batch is None:
        raise IndexError(f"sample-index {args.sample_index} is out of range for the {args.split} dataloader")

    logger.info("=== One Batch Overview ===")
    logger.info("Batch keys: %s", list(batch.keys()))
    for key, value in batch.items():
        logger.info("  %s: %s", key, _shape_of(value))

    device = torch.device(args.device)
    batch = _to_device(batch, device)

    logger.info("=== Forward Flow ===")
    with torch.no_grad():
        outputs = model(return_loss=True, **batch)

    logger.info("Forward output keys: %s", list(outputs.keys()))
    for key, value in outputs.items():
        logger.info("  %s: %s", key, _shape_of(value))

    logger.info("Done. The log file now contains the module tree and tensor flow.")


if __name__ == "__main__":
    main()
