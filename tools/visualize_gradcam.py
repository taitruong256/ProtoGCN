import argparse
import logging
import os
from pathlib import Path

from mmcv import Config

from protogcn.apis import init_recognizer, visualize_gradcam_test_loader
from protogcn.datasets import build_dataset, build_dataloader


REPO_ROOT = Path(__file__).resolve().parents[1]


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
    parser.add_argument("--config", default=None, help="Path to config file")
    parser.add_argument("--checkpoint", default=None, help="Path to checkpoint; defaults to latest .pth in work_dirs")
    parser.add_argument("--output-dir", default=None, help="Directory to save PNG/GIF outputs")
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


def _config_has_model(cfg):
    try:
        return cfg.get("model", None) is not None
    except Exception:
        return False


def _config_hints(path):
    path = Path(path)
    dataset = None
    parts = path.parts
    for idx, part in enumerate(parts):
        if part in {"configs", "work_dirs"} and idx + 1 < len(parts):
            dataset = parts[idx + 1]
            break
    stem = path.stem
    base_stem = stem.split("_")[0]
    return dataset, stem, base_stem


def _score_config_candidate(candidate, initial_path=None, checkpoint_dir=None):
    score = 0
    candidate = Path(candidate)
    if candidate.suffix != ".py":
        return None

    if initial_path is not None and candidate.resolve() == Path(initial_path).resolve():
        score -= 100

    if checkpoint_dir is not None:
        try:
            if candidate.parent.resolve() == Path(checkpoint_dir).resolve():
                score -= 20
        except Exception:
            pass

    dataset_hint = stem_hint = base_hint = None
    if initial_path is not None:
        dataset_hint, stem_hint, base_hint = _config_hints(initial_path)

    if dataset_hint and dataset_hint in candidate.parts:
        score -= 8
    if stem_hint and candidate.stem == stem_hint:
        score -= 12
    elif stem_hint and candidate.stem.startswith(stem_hint):
        score -= 8
    elif base_hint and candidate.stem.startswith(base_hint):
        score -= 4

    if "configs" in candidate.parts:
        score -= 2
    if "work_dirs" in candidate.parts:
        score += 1
    return score


def _iter_config_candidates(initial_config=None, checkpoint_dir=None):
    seen = set()
    for root in [Path(initial_config).parent if initial_config else None, Path(checkpoint_dir) if checkpoint_dir else None,
                 REPO_ROOT / "configs", REPO_ROOT / "work_dirs"]:
        if root is None or not root.exists():
            continue
        for candidate in root.rglob("*.py"):
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            yield candidate


def _find_effective_config(initial_config=None, checkpoint_dir=None):
    """Find a config file that actually defines `model`."""
    candidates = []
    for candidate in _iter_config_candidates(initial_config, checkpoint_dir):
        score = _score_config_candidate(candidate, initial_path=initial_config, checkpoint_dir=checkpoint_dir)
        if score is None:
            continue
        candidates.append((score, candidate))

    candidates.sort(key=lambda item: item[0])
    for _, candidate in candidates:
        try:
            cfg = Config.fromfile(str(candidate))
        except Exception as exc:
            logging.getLogger(__name__).debug("Skipping config %s: %s", candidate, exc)
            continue
        if _config_has_model(cfg):
            return str(candidate), cfg

    searched = [str(candidate) for _, candidate in candidates]
    raise ValueError(
        f"Could not find a usable config with a top-level `model` key. "
        f"Searched: {searched}"
    )


def _discover_checkpoint():
    work_dirs = REPO_ROOT / "work_dirs"
    ckpts = [p for p in work_dirs.rglob("*.pth") if p.is_file()]
    if not ckpts:
        raise FileNotFoundError(f"No .pth checkpoint found under {work_dirs}")
    ckpts = sorted(ckpts, key=lambda p: p.stat().st_mtime)
    return str(ckpts[-1])


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logger = logging.getLogger(__name__)

    logger.info("Loading config")
    requested_config = args.config
    requested_checkpoint = args.checkpoint

    logger.info("Resolving checkpoint and output directory")
    checkpoint = requested_checkpoint or _discover_checkpoint()
    checkpoint = str(Path(checkpoint).resolve())
    if not checkpoint.endswith(".pth"):
        checkpoint = _latest_checkpoint(Path(checkpoint).parent)
    checkpoint_dir = Path(checkpoint).parent

    if requested_config is not None and Path(requested_config).exists():
        cfg = Config.fromfile(str(requested_config))
    else:
        cfg = None

    if cfg is None or not _config_has_model(cfg):
        effective_config, cfg = _find_effective_config(requested_config, checkpoint_dir)
        logger.info("Using config with model: %s", effective_config)
    else:
        logger.info("Using config from argument: %s", requested_config)

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
