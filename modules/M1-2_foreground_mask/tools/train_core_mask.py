from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geocore_mask.models.registry import build_model
from geocore_mask.utils.metrics import binary_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune the M1-2 core foreground mask UNet.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--base-package", default="models/core_mask_unet_v1")
    parser.add_argument("--output-package", default="models/core_mask_unet_v2")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--steps-per-epoch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def resolve_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_split(split_path: Path, dataset_root: Path | None = None) -> list[tuple[Path, Path]]:
    samples: list[tuple[Path, Path]] = []
    for line in split_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        image_rel, mask_rel = line.split()[:2]
        image_path = Path(image_rel)
        mask_path = Path(mask_rel)
        if not image_path.is_absolute() and dataset_root is not None:
            image_path = dataset_root / image_path
        else:
            image_path = resolve_path(image_path)
        if not mask_path.is_absolute() and dataset_root is not None:
            mask_path = dataset_root / mask_path
        else:
            mask_path = resolve_path(mask_path)
        samples.append((image_path.resolve(), mask_path.resolve()))
    return samples


def load_image(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    return arr


def load_mask(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        arr = np.asarray(img.convert("L"), dtype=np.float32)
    return (arr > 127).astype(np.float32)


class CoreMaskCropDataset(Dataset):
    def __init__(
        self,
        samples: list[tuple[Path, Path]],
        *,
        image_size: int,
        mean: list[float],
        std: list[float],
        steps_per_epoch: int,
        augment: bool,
    ):
        self.samples = samples
        self.image_size = image_size
        self.mean = np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.asarray(std, dtype=np.float32).reshape(1, 1, 3)
        self.std[self.std == 0] = 1.0
        self.steps_per_epoch = max(1, steps_per_epoch)
        self.augment = augment

    def __len__(self) -> int:
        return self.steps_per_epoch

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, mask_path = random.choice(self.samples)
        image = load_image(image_path)
        mask = load_mask(mask_path)
        image, mask = self._crop(image, mask)

        if self.augment:
            if random.random() < 0.5:
                image = np.flip(image, axis=1).copy()
                mask = np.flip(mask, axis=1).copy()
            if random.random() < 0.5:
                image = np.flip(image, axis=0).copy()
                mask = np.flip(mask, axis=0).copy()

        image = (image - self.mean) / self.std
        bg = 1.0 - mask
        target = np.stack([bg, mask], axis=0).astype(np.float32)
        tensor = torch.from_numpy(image.transpose(2, 0, 1).astype(np.float32))
        return tensor, torch.from_numpy(target)

    def _crop(self, image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        size = self.image_size
        h, w = mask.shape
        pad_h = max(0, size - h)
        pad_w = max(0, size - w)
        if pad_h or pad_w:
            image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
            mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode="edge")
            h, w = mask.shape

        fg = np.argwhere(mask > 0.5)
        if len(fg) and random.random() < 0.8:
            cy, cx = fg[random.randrange(len(fg))]
            y0 = int(np.clip(cy - random.randrange(size), 0, h - size))
            x0 = int(np.clip(cx - random.randrange(size), 0, w - size))
        else:
            y0 = random.randrange(0, h - size + 1)
            x0 = random.randrange(0, w - size + 1)
        return image[y0 : y0 + size, x0 : x0 + size], mask[y0 : y0 + size, x0 : x0 + size]


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)


def bce_dice_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy(pred, target)
    pred_fg = pred[:, 1]
    target_fg = target[:, 1]
    smooth = 1.0
    intersection = (pred_fg * target_fg).sum(dim=(1, 2))
    dice = (2 * intersection + smooth) / (pred_fg.sum(dim=(1, 2)) + target_fg.sum(dim=(1, 2)) + smooth)
    return bce + (1.0 - dice.mean())


@torch.no_grad()
def evaluate(model, samples: list[tuple[Path, Path]], *, image_size: int, mean: list[float], std: list[float], device: torch.device) -> dict:
    if not samples:
        return {}
    dataset = CoreMaskCropDataset(
        samples,
        image_size=image_size,
        mean=mean,
        std=std,
        steps_per_epoch=max(8, len(samples) * 8),
        augment=False,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    metrics_accum: list[dict[str, float]] = []
    model.eval()
    for image, target in loader:
        image = image.to(device)
        target_np = target[:, 1].numpy() > 0.5
        pred = model(image)
        if isinstance(pred, (tuple, list)):
            pred = pred[0]
        pred_np = (pred[:, 1].detach().cpu().numpy() >= 0.5)
        metrics_accum.append(binary_metrics(pred_np, target_np))
    return {key: float(np.mean([m[key] for m in metrics_accum])) for key in metrics_accum[0]}


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    config_path = resolve_path(args.config)
    config = load_json(config_path)
    base_package = resolve_path(args.base_package)
    output_package = resolve_path(args.output_package)
    base_manifest = load_json(base_package / "model_manifest.json")

    data_cfg = config["data"]
    dataset_root = resolve_path(data_cfg["dataset_root"]) if data_cfg.get("dataset_root") else None
    train_samples = load_split(resolve_path(data_cfg["train_list"]), dataset_root)
    val_samples = load_split(resolve_path(data_cfg.get("val_list", data_cfg["train_list"])), dataset_root)
    test_samples = load_split(resolve_path(data_cfg.get("test_list", data_cfg["train_list"])), dataset_root)
    if not val_samples:
        val_samples = test_samples or train_samples

    train_all = list(dict.fromkeys(train_samples + val_samples + test_samples))
    if len(train_all) <= 3:
        train_samples = train_all

    training_cfg = config.get("training", {})
    image_size = int(data_cfg.get("image_size", base_manifest.get("input", {}).get("image_size", 512)))
    epochs = int(args.epochs or training_cfg.get("epochs", 100))
    batch_size = int(args.batch_size or training_cfg.get("batch_size", 4))
    lr = float(args.lr or training_cfg.get("learning_rate", 1e-3))

    input_cfg = base_manifest.get("input", {})
    mean = input_cfg.get("mean", [0.0, 0.0, 0.0])
    std = input_cfg.get("std", [1.0, 1.0, 1.0])
    device = resolve_device(args.device)

    model = build_model(base_manifest.get("model_name", "CoreMaskUNet"), input_channels=3, num_classes=2)
    weights_path = base_package / str(base_manifest["weights"])
    state = torch.load(weights_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.to(device)

    dataset = CoreMaskCropDataset(
        train_samples,
        image_size=image_size,
        mean=mean,
        std=std,
        steps_per_epoch=args.steps_per_epoch,
        augment=True,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=args.num_workers)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses: list[float] = []
        for image, target in loader:
            image = image.to(device)
            target = target.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(image)
            if isinstance(pred, (tuple, list)):
                pred = pred[0]
            loss = bce_dice_loss(pred, target)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        val_metrics = evaluate(model, val_samples, image_size=image_size, mean=mean, std=std, device=device)
        row = {"epoch": epoch, "loss": float(np.mean(losses)), **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))

    output_package.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_package / "weights.pth")

    manifest = dict(base_manifest)
    manifest["model_version"] = "v2.0.0-finetuned"
    manifest["weights"] = "weights.pth"
    manifest["training"] = {
        "source_config": str(config_path),
        "base_package": str(base_package),
        "train_sample_count": len(train_samples),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "steps_per_epoch": args.steps_per_epoch,
        "device": str(device),
    }
    (output_package / "model_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    model_card = output_package / "model_card.md"
    model_card.write_text(
        "# CoreMaskUNet v2\n\n"
        "Fine-tuned from `core_mask_unet_v1` using ArcGIS shapefile annotations in `datasets/core_mask_v2`.\n\n"
        f"- epochs: {epochs}\n"
        f"- train samples: {len(train_samples)}\n"
        f"- image size: {image_size}\n"
        f"- device: {device}\n",
        encoding="utf-8",
    )
    (output_package / "train_history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[ok] wrote model package: {output_package}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
