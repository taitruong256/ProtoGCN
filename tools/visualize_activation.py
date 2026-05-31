import argparse
import glob
import logging
import os

from mmcv import Config

from protogcn.apis import init_recognizer, visualize_test_loader
from protogcn.datasets import build_dataset, build_dataloader


def _latest_checkpoint(work_dir):
    ckpts = sorted(glob.glob(os.path.join(work_dir, "*.pth")))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint found in {work_dir}")
    return ckpts[-1]


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize gait activation as GIFs")
    parser.add_argument("--config", required=True, help="Path to config file")
    parser.add_argument("--checkpoint", default=None, help="Path to checkpoint; defaults to latest .pth in work_dir")
    parser.add_argument("--output-dir", default=None, help="Directory to save PNG/GIF outputs")
    parser.add_argument("--device", default="cuda:0", help="Torch device, e.g. cuda:0 or cpu")
    parser.add_argument("--class-mode", default="pred", choices=["pred", "max"])
    parser.add_argument("--no-gif", action="store_true", help="Only save PNG frames")
    return parser.parse_args()


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
    output_dir = args.output_dir or os.path.join(cfg.work_dir, "visualize")

    logger.info("Loading model and building test loader")
    model = init_recognizer(cfg, checkpoint=checkpoint, device=args.device)
    dataset = build_dataset(cfg.data.test, dict(test_mode=True))
    loader = build_dataloader(
        dataset,
        videos_per_gpu=1,
        workers_per_gpu=0,
        shuffle=False,
    )

    logger.info("Starting visualization")
    visualize_test_loader(
        model,
        loader,
        output_dir=output_dir,
        class_mode=args.class_mode,
        render_gif=not args.no_gif,
    )
    logger.info("Visualization finished. Output saved to %s", output_dir)


if __name__ == "__main__":
    main()
