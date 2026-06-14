import argparse
import glob
import logging
import os
from pathlib import Path

from mmcv import Config

from protogcn.apis import init_recognizer, visualize_gradcam_test_loader
from protogcn.datasets import build_dataset, build_dataloader


def _latest_checkpoint(work_dir):
    work_path = Path(work_dir)
    if work_path.is_file():
        if work_path.suffix == ".pth":
            return str(work_path)
        work_path = work_path.parent

    ckpts = [p for p in work_path.glob("*.pth") if p.is_file()]
    if not ckpts:
        raise FileNotFoundError(f"No .pth checkpoint found in {work_path}")
    ckpts = sorted(ckpts, key=lambda p: p.stat().st_mtime)
    return str(ckpts[-1])


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize gait Grad-CAM as GIFs")
    parser.add_argument("--config", default='/home/taitruong256/taitruong/CCU/GaitXplain/docs/ProtoGCN_gait/work_dirs/casia_b/checkpoints_casia_b/j_3/j_2.py', help="Path to config file")
    parser.add_argument("--checkpoint", default='/home/taitruong256/taitruong/CCU/GaitXplain/docs/ProtoGCN_gait/work_dirs/casia_b/checkpoints_casia_b/j_3/best_gait_contrastive_loss_epoch_144.pth', help="Path to checkpoint; defaults to latest .pth in work_dir")
    parser.add_argument("--output-dir", default='data/output/', help="Directory to save PNG/GIF outputs")
    parser.add_argument("--device", default="cuda:0", help="Torch device, e.g. cuda:0 or cpu")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"], help="Dataset split to visualize")
    parser.add_argument("--class-mode", default="pred", choices=["pred", "max"])
    parser.add_argument("--no-gif", action="store_true", help="Only save PNG frames")
    return parser.parse_args()


def _build_split_dataset(cfg, split):
    if not hasattr(cfg.data, split):
        raise AttributeError(f"Config does not define data.{split}")
    dataset_cfg = getattr(cfg.data, split)
    return build_dataset(dataset_cfg, dict(test_mode=(split != "train")))


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logger = logging.getLogger(__name__)

    logger.info("Loading config")
    cfg = Config.fromfile(args.config)

    logger.info("Resolving checkpoint and output directory")
    checkpoint = args.checkpoint or _latest_checkpoint(cfg.work_dir)
    checkpoint = str(Path(checkpoint).resolve())
    if not checkpoint.endswith(".pth"):
        checkpoint = _latest_checkpoint(Path(checkpoint).parent)
    checkpoint_dir = Path(checkpoint).parent

    workdir_configs = sorted(checkpoint_dir.glob("*.py"))
    effective_config = None
    for candidate in workdir_configs:
        if candidate.resolve() != Path(args.config).resolve():
            effective_config = candidate
            break

    if effective_config is not None:
        logger.info("Using config from work_dir: %s", effective_config)
        cfg = Config.fromfile(str(effective_config))
    else:
        logger.info("Using config from argument: %s", args.config)

    cfg.model = cfg.model.copy()
    cfg.model.pop("view_loss_weight", None)

    output_dir = args.output_dir or os.path.join(cfg.work_dir, f"visualize_gradcam_{args.split}")

    logger.info("Using checkpoint: %s", checkpoint)

    logger.info("Loading model and building %s loader", args.split)
    model = init_recognizer(cfg, checkpoint=checkpoint, device=args.device)
    dataset = _build_split_dataset(cfg, args.split)
    loader = build_dataloader(
        dataset,
        videos_per_gpu=1,
        workers_per_gpu=0,
        shuffle=False,
    )

    logger.info("Starting visualization")
    visualize_gradcam_test_loader(
        model,
        loader,
        output_dir=output_dir,
        class_mode=args.class_mode,
        render_gif=not args.no_gif,
        save_heatmap=True,
    )
    logger.info("Visualization finished. Output saved to %s", output_dir)


if __name__ == "__main__":
    main()
