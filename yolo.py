"""Prepare CCPD2019 for Ultralytics YOLO26 and train a plate detector.

CCPD encodes the plate bounding box in every image filename.  This script
converts that box to YOLO's normalized ``class x_center y_center width height``
format without duplicating the (large) image set: the generated image folders
contain symbolic links to ``CCPD2019/ccpd_base``.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import struct
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CCPD_ROOT = ROOT / "CCPD2019"
SOURCE_IMAGES = CCPD_ROOT / "ccpd_base"
SPLITS_DIR = CCPD_ROOT / "splits"
YOLO_DATASET = CCPD_ROOT / "yolo_ccpd_base"
RUNS_DIR = ROOT / "runs" / "license_plate"
EXPORTED_WEIGHTS = ROOT / "weights" / "yolo26n_ccpd_best.pt"
VAL_LIMIT = 10_000


def jpeg_size(path: Path) -> tuple[int, int]:
    """Read JPEG width and height from its header without decoding the image."""
    sof_markers = {
        0xC0,
        0xC1,
        0xC2,
        0xC3,
        0xC5,
        0xC6,
        0xC7,
        0xC9,
        0xCA,
        0xCB,
        0xCD,
        0xCE,
        0xCF,
    }
    with path.open("rb") as stream:
        if stream.read(2) != b"\xff\xd8":
            raise ValueError(f"不是有效的 JPEG 文件: {path}")
        while True:
            byte = stream.read(1)
            if not byte:
                break
            if byte != b"\xff":
                continue
            while byte == b"\xff":
                byte = stream.read(1)
            marker = byte[0]
            if marker in {0x01, *range(0xD0, 0xDA)}:
                continue
            raw_length = stream.read(2)
            if len(raw_length) != 2:
                break
            segment_length = struct.unpack(">H", raw_length)[0]
            if segment_length < 2:
                break
            if marker in sof_markers:
                data = stream.read(5)
                if len(data) != 5:
                    break
                height, width = struct.unpack(">HH", data[1:])
                return width, height
            stream.seek(segment_length - 2, os.SEEK_CUR)
    raise ValueError(f"无法从 JPEG 头读取尺寸: {path}")


def read_official_split(name: str) -> list[Path]:
    split_file = SPLITS_DIR / f"{name}.txt"
    if not split_file.is_file():
        raise FileNotFoundError(f"缺少数据集划分文件: {split_file}")

    result: list[Path] = []
    seen: set[Path] = set()
    for line_number, line in enumerate(
        split_file.read_text(encoding="utf-8").splitlines(), start=1
    ):
        value = line.strip().replace("\\", "/")
        if not value:
            continue
        relative = Path(value)
        # Official files contain ccpd_base/<filename>; reject paths outside it.
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"{split_file}:{line_number} 含有非法路径: {value}")
        image = CCPD_ROOT / relative
        if image.parent.resolve() != SOURCE_IMAGES.resolve() or not image.is_file():
            raise FileNotFoundError(f"{split_file}:{line_number} 图片不存在: {image}")
        if image not in seen:
            result.append(image)
            seen.add(image)
    return result


def plate_box_from_filename(path: Path) -> tuple[int, int, int, int]:
    """Return CCPD's axis-aligned plate box (left, top, right, bottom)."""
    try:
        box_field = path.stem.split("-")[2]
        top_left, bottom_right = box_field.split("_")
        left, top = (int(value) for value in top_left.split("&"))
        right, bottom = (int(value) for value in bottom_right.split("&"))
    except (IndexError, ValueError) as error:
        raise ValueError(f"无法从 CCPD 文件名解析边界框: {path.name}") from error
    if right <= left or bottom <= top:
        raise ValueError(f"CCPD 边界框无效: {path.name}")
    return left, top, right, bottom


def yolo_label(path: Path) -> str:
    if path.suffix.lower() not in {".jpg", ".jpeg"}:
        raise ValueError(f"当前 CCPD 转换器只支持 JPEG: {path}")
    image_width, image_height = jpeg_size(path)
    left, top, right, bottom = plate_box_from_filename(path)
    left = min(max(left, 0), image_width - 1)
    top = min(max(top, 0), image_height - 1)
    right = min(max(right, left + 1), image_width)
    bottom = min(max(bottom, top + 1), image_height)
    x_center = (left + right) / (2 * image_width)
    y_center = (top + bottom) / (2 * image_height)
    width = (right - left) / image_width
    height = (bottom - top) / image_height
    return f"0 {x_center:.8f} {y_center:.8f} {width:.8f} {height:.8f}\n"


def ensure_image_link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        if destination.resolve() == source.resolve():
            return
        destination.unlink()
    elif destination.exists():
        raise FileExistsError(f"目标位置已有非符号链接文件: {destination}")
    target = os.path.relpath(source, start=destination.parent)
    destination.symlink_to(target)


def write_split(name: str, images: list[Path]) -> None:
    image_dir = YOLO_DATASET / "images" / name
    label_dir = YOLO_DATASET / "labels" / name
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    expected_stems = {image.stem for image in images}
    # Remove leftovers when seed/test ratio changes between invocations.
    for directory in (image_dir, label_dir):
        for old_path in directory.iterdir():
            if old_path.stem not in expected_stems:
                old_path.unlink()

    for index, source in enumerate(images, start=1):
        ensure_image_link(source, image_dir / source.name)
        (label_dir / f"{source.stem}.txt").write_text(
            yolo_label(source), encoding="utf-8"
        )
        if index % 20_000 == 0 or index == len(images):
            print(f"[{name}] 已转换 {index:,}/{len(images):,}")


def prepare_dataset(test_ratio: float, seed: int) -> Path:
    if not 0.0 < test_ratio < 1.0:
        raise ValueError("--test-ratio 必须在 0 和 1 之间")
    if not SOURCE_IMAGES.is_dir():
        raise FileNotFoundError(f"缺少 CCPD 图片目录: {SOURCE_IMAGES}")

    train = read_official_split("train")
    official_val = read_official_split("val")
    overlap = set(train).intersection(official_val)
    if overlap:
        raise ValueError(f"官方 train/val 存在 {len(overlap)} 张重复图片")

    val = official_val[:VAL_LIMIT]
    all_images = train + official_val
    test_count = max(1, round(len(all_images) * test_ratio))
    # Protect the requested validation subset and sample test from the remaining
    # ccpd_base images. Any sampled official-train images are removed from train.
    test_candidates = sorted(set(all_images).difference(val))
    test_set = set(random.Random(seed).sample(test_candidates, test_count))
    splits = {
        "train": [image for image in train if image not in test_set],
        "val": val,
        "test": sorted(test_set),
    }
    for name, images in splits.items():
        write_split(name, images)

    dataset_yaml = YOLO_DATASET / "ccpd.yaml"
    dataset_yaml.write_text(
        f"path: {YOLO_DATASET.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        "  0: license_plate\n",
        encoding="utf-8",
    )
    metadata = {
        "seed": seed,
        "test_ratio": test_ratio,
        "counts": {name: len(images) for name, images in splits.items()},
    }
    (YOLO_DATASET / "split_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"YOLO 数据集配置已生成: {dataset_yaml}")
    print(f"数据量: {metadata['counts']}")
    return dataset_yaml


def train(args: argparse.Namespace, dataset_yaml: Path) -> Path:
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError("缺少 ultralytics，请先执行 `uv sync`") from error

    model = YOLO(args.model)
    model.train(
        data=str(dataset_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=-1,  # Let Ultralytics target about 60% of available GPU memory.
        amp=True,
        patience=args.patience,
        workers=args.workers,
        device=args.device,
        project=str(RUNS_DIR),
        name=args.run_name,
        exist_ok=args.exist_ok,
        save=True,
        seed=args.seed,
        deterministic=True,
        plots=True,
    )

    best = Path(model.trainer.best).resolve()
    if not best.is_file():
        raise FileNotFoundError(f"训练结束但未找到 best.pt: {best}")
    EXPORTED_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, EXPORTED_WEIGHTS)
    print(f"最优模型权重已保存: {EXPORTED_WEIGHTS}")

    if not args.skip_test:
        validation_batch = max(1, int(model.trainer.args.batch))
        YOLO(str(EXPORTED_WEIGHTS)).val(
            data=str(dataset_yaml),
            split="test",
            imgsz=args.imgsz,
            batch=validation_batch,
        )
    return EXPORTED_WEIGHTS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="yolo26n.pt", help="预训练权重")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--device", default=None, help="例如 0、cpu；默认自动选择")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--run-name", default="yolo26n_ccpd")
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--skip-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_yaml = prepare_dataset(args.test_ratio, args.seed)
    if not args.prepare_only:
        train(args, dataset_yaml)


if __name__ == "__main__":
    main()
