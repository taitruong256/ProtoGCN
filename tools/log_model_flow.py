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


def _first_tensor_shape(obj):
    if torch.is_tensor(obj):
        return tuple(obj.shape)
    if isinstance(obj, (list, tuple)):
        for item in obj:
            shape = _first_tensor_shape(item)
            if shape is not None:
                return shape
    if isinstance(obj, dict):
        for item in obj.values():
            shape = _first_tensor_shape(item)
            if shape is not None:
                return shape
    return None


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


def _gcn_tcn_flow_section():
    lines = []
    lines.append("```mermaid")
    lines.append("flowchart TD")
    lines.append("    X[Input x: N x C x T x V]")
    lines.append("    R[Residual branch]")
    lines.append("    G[unit_gcn]")
    lines.append("    T[mstcn]")
    lines.append("    A[Add residual]")
    lines.append("    Y[Output y]")
    lines.append("")
    lines.append("    X --> R")
    lines.append("    X --> G")
    lines.append("    G --> T")
    lines.append("    T --> A")
    lines.append("    R --> A")
    lines.append("    A --> Y")
    lines.append("")
    lines.append("    subgraph GCN[unit_gcn]")
    lines.append("        G1[Learn / update adjacency]")
    lines.append("        G2[Spatial aggregation over joints V]")
    lines.append("        G3[Return x_gcn and graph]")
    lines.append("        G1 --> G2 --> G3")
    lines.append("    end")
    lines.append("")
    lines.append("    subgraph TCN[mstcn]")
    lines.append("        T1[Multi-scale temporal branches]")
    lines.append("        T2[Concat branches on channel dim]")
    lines.append("        T3[Temporal mixing + BN + Dropout]")
    lines.append("        T1 --> T2 --> T3")
    lines.append("    end")
    lines.append("```")
    lines.append("")
    lines.append("- `unit_gcn` learns spatial relationships between joints using the graph adjacency.")
    lines.append("- `mstcn` models temporal evolution across frames, usually keeping `V` fixed and changing `T` only when `stride > 1`.")
    lines.append("- The residual path preserves the original signal so the block behaves like `mstcn(unit_gcn(x)) + residual(x)` before `ReLU`.")
    return lines


def _register_size_hooks(model):
    records = []
    handles = []

    backbone = getattr(model, "backbone", None)
    if backbone is not None and hasattr(backbone, "gcn"):
        for idx, block in enumerate(backbone.gcn):
            def _make_block_hook(block_idx):
                def _hook(module, inputs, output):
                    out_tensor = output[0] if isinstance(output, (tuple, list)) else output
                    records.append({
                        "name": f"GCN Block {block_idx + 1}",
                        "input": _first_tensor_shape(inputs[0]),
                        "output": _first_tensor_shape(out_tensor),
                    })
                return _hook

            handles.append(block.register_forward_hook(_make_block_hook(idx)))

    cls_head = getattr(model, "cls_head", None)
    if cls_head is not None:
        def _cls_hook(module, inputs, output):
            records.append({
                "name": "Classifier",
                "input": _first_tensor_shape(inputs[0]),
                "output": _first_tensor_shape(output),
            })

        handles.append(cls_head.register_forward_hook(_cls_hook))

    return handles, records


def _format_size(shape):
    if shape is None:
        return "?"
    return " x ".join(str(dim) for dim in shape)


def _size_flow_section(records):
    lines = []
    lines.append("```mermaid")
    lines.append("flowchart LR")
    for i, record in enumerate(records):
        node_in = f"N{i}_IN"
        node_out = f"N{i}_OUT"
        label = record["name"]
        lines.append(f'    {node_in}["{label} in\\n{_format_size(record["input"])}"]')
        lines.append(f'    {node_out}["{label} out\\n{_format_size(record["output"])}"]')
        lines.append(f"    {node_in} --> {node_out}")
        if i < len(records) - 1:
            lines.append(f"    {node_out} --> N{i+1}_IN")
    lines.append("```")
    lines.append("")
    lines.append("### Table")
    for record in records:
        lines.append(f"- {record['name']}: `{_format_size(record['input'])}` -> `{_format_size(record['output'])}`")
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
    lines.append("## 3. GCN/TCN Flow")
    lines.extend(_gcn_tcn_flow_section())
    lines.append("## 4. Module Details")
    lines.extend(_module_details_section(model))
    lines.append("## 5. Full Model Detail")
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

    size_handles, size_records = _register_size_hooks(model)
    logger.info("=== Forward Flow ===")
    with torch.no_grad():
        outputs = model(return_loss=True, **batch)
    for handle in size_handles:
        handle.remove()

    logger.info("=== Size Flow ===")
    for record in size_records:
        logger.info(
            "%s: %s -> %s",
            record["name"],
            _format_size(record["input"]),
            _format_size(record["output"]),
        )

    logger.info("Forward output keys: %s", list(outputs.keys()))
    for key, value in outputs.items():
        logger.info("  %s: %s", key, _shape_of(value))

    logger.info("Done. The log file now contains the module tree and tensor flow.")


if __name__ == "__main__":
    main()
