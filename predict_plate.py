"""Evaluate license plate detection and OCR on the university_plates dataset.

Pipeline:

1. Load ``weights/yolo26n_ccpd_best.pt``.
2. Run YOLO on four university_plates subsets:
   * plates
   * night_plates
   * multi_plates
   * upward_downward_plates
3. Crop every detected plate box and recognize it with PaddleOCR
   TextRecognition.
4. Save annotated images, per-image CSV results, and metrics.

The accuracy metrics stay the same as before:

* Full Match Accuracy
* Character-level Accuracy

The university_plates image filenames do not contain ground-truth plate
numbers, and no annotation file is currently present in the dataset directory.
Therefore the script always reports detection/OCR outputs, and computes the two
accuracy metrics only when ground truth is available through ``--truth-csv`` or
CCPD-style filenames.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import os
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = ROOT / "weights" / "yolo26n_ccpd_best.pt"
DEFAULT_DATASET_ROOT = ROOT / "university_plates"
DEFAULT_OUTPUT = ROOT / "runs" / "plate_ocr_eval" / "university_plates"
DEFAULT_FONT = Path("/usr/share/fonts/adobe-source-han-sans/SourceHanSansCN-Regular.otf")
SUBSETS = ("plates", "night_plates", "multi_plates", "upward_downward_plates")

PROVINCES = [
    "皖",
    "沪",
    "津",
    "渝",
    "冀",
    "晋",
    "蒙",
    "辽",
    "吉",
    "黑",
    "苏",
    "浙",
    "京",
    "闽",
    "赣",
    "鲁",
    "豫",
    "鄂",
    "湘",
    "粤",
    "桂",
    "琼",
    "川",
    "贵",
    "云",
    "藏",
    "陕",
    "甘",
    "青",
    "宁",
    "新",
    "警",
    "学",
    "O",
]
ALPHABETS = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "J",
    "K",
    "L",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
    "O",
]
ADS = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "J",
    "K",
    "L",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "O",
]


@dataclass(frozen=True)
class DatasetImage:
    path: Path
    subset: str


@dataclass
class PlateDetection:
    pred: str
    confidence: float
    box_xyxy: tuple[int, int, int, int]


@dataclass
class ImageResult:
    image: Path
    subset: str
    gt_plates: list[str]
    detections: list[PlateDetection]
    char_distance: int | None
    full_match: bool | None


def require_cv2() -> Any:
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("缺少 opencv-python，请先执行 `uv sync` 安装依赖") from error
    return cv2


def normalize_plate(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).upper()
    return "".join(re.findall(r"[\u4e00-\u9fffA-Z0-9]", text))


def split_plate_text(value: str) -> list[str]:
    """Split one or more plate numbers from CSV text."""
    if not value:
        return []
    values = re.split(r"[|,，;；\s]+", value.strip())
    return [normalized for item in values if (normalized := normalize_plate(item))]


def decode_ccpd_plate_from_filename(path: Path) -> list[str]:
    """Decode CCPD-style filenames when they are used as an optional test set."""
    try:
        encoded = path.stem.split("-")[4]
        indexes = [int(value) for value in encoded.split("_")]
    except (IndexError, ValueError):
        return []

    if len(indexes) != 7:
        return []
    try:
        return [
            PROVINCES[indexes[0]]
            + ALPHABETS[indexes[1]]
            + "".join(ADS[index] for index in indexes[2:])
        ]
    except IndexError:
        return []


def edit_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)

    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        for j, right_char in enumerate(right, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def plate_list_distance(gt_plates: list[str], pred_plates: list[str]) -> int:
    """Greedily match predicted plates to GT plates by minimum edit distance."""
    remaining = pred_plates.copy()
    total = 0
    for gt in gt_plates:
        if not remaining:
            total += len(gt)
            continue
        best_index, best_distance = min(
            enumerate(edit_distance(gt, pred) for pred in remaining),
            key=lambda item: item[1],
        )
        total += best_distance
        remaining.pop(best_index)
    return total


def full_match(gt_plates: list[str], pred_plates: list[str]) -> bool:
    return sorted(gt_plates) == sorted(pred_plates)


def accepted_kwargs(function: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return kwargs
    if any(param.kind == param.VAR_KEYWORD for param in signature.parameters.values()):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in signature.parameters}


def build_paddle_recognizer(device: str | None) -> Any:
    """Create a PaddleOCR recognizer for already-cropped plate images."""
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    os.environ.setdefault("FLAGS_use_onednn", "0")
    try:
        from paddleocr import PaddleOCR, TextRecognition
    except ImportError as error:
        raise RuntimeError("缺少 paddleocr，请先执行 `uv sync` 安装依赖") from error

    recognition_kwargs: dict[str, Any] = {}
    if device is not None:
        recognition_kwargs["device"] = device
    try:
        return TextRecognition(
            **accepted_kwargs(TextRecognition.__init__, recognition_kwargs)
        )
    except Exception:
        pass

    fallback_kwargs = {
        "lang": "ch",
        "device": device,
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "use_textline_orientation": True,
    }
    try:
        return PaddleOCR(
            **{
                key: value
                for key, value in accepted_kwargs(
                    PaddleOCR.__init__, fallback_kwargs
                ).items()
                if value is not None
            }
        )
    except Exception as error:
        raise RuntimeError(f"初始化 PaddleOCR 识别器失败: {error}") from error


def extract_texts_from_ocr_result(result: Any) -> list[tuple[str, float]]:
    texts: list[tuple[str, float]] = []

    def walk(value: Any) -> None:
        if value is None:
            return
        if hasattr(value, "to_dict"):
            try:
                walk(value.to_dict())
                return
            except Exception:
                pass
        if hasattr(value, "json"):
            try:
                json_value = value.json
                walk(json_value() if callable(json_value) else json_value)
                return
            except Exception:
                pass
        if hasattr(value, "res"):
            try:
                walk(value.res)
                return
            except Exception:
                pass

        if isinstance(value, dict):
            rec_text = value.get("rec_text") or value.get("text")
            if isinstance(rec_text, str):
                score = value.get("rec_score") or value.get("score") or 0.0
                texts.append((rec_text, float(score or 0.0)))
                return
            rec_texts = value.get("rec_texts") or value.get("texts")
            rec_scores = value.get("rec_scores") or value.get("scores") or []
            if isinstance(rec_texts, list):
                for index, text in enumerate(rec_texts):
                    if isinstance(text, str):
                        score = rec_scores[index] if index < len(rec_scores) else 0.0
                        texts.append((text, float(score or 0.0)))
                return
            for child in value.values():
                walk(child)
            return

        if isinstance(value, tuple) and len(value) >= 2 and isinstance(value[0], str):
            try:
                texts.append((value[0], float(value[1] or 0.0)))
            except (TypeError, ValueError):
                texts.append((value[0], 0.0))
            return

        if (
            isinstance(value, (list, tuple))
            and len(value) >= 2
            and isinstance(value[1], tuple)
            and len(value[1]) >= 2
            and isinstance(value[1][0], str)
        ):
            try:
                texts.append((value[1][0], float(value[1][1] or 0.0)))
            except (TypeError, ValueError):
                texts.append((value[1][0], 0.0))
            return

        if isinstance(value, (list, tuple)):
            for child in value:
                walk(child)

    walk(result)
    return [(text, score) for text, score in texts if normalize_plate(text)]


def recognize_crop(ocr: Any, crop_bgr: Any) -> str:
    if crop_bgr.size == 0:
        return ""

    results: list[Any] = []
    if hasattr(ocr, "predict"):
        try:
            results.append(ocr.predict(input=crop_bgr))
        except TypeError:
            results.append(ocr.predict(crop_bgr))
        except Exception:
            pass
    if hasattr(ocr, "ocr"):
        try:
            results.append(ocr.ocr(crop_bgr))
        except Exception:
            pass

    candidates: list[tuple[str, float]] = []
    for result in results:
        candidates.extend(extract_texts_from_ocr_result(result))
    if not candidates:
        return ""
    return normalize_plate("".join(text for text, _score in candidates))


def clip_box(
    box_xyxy: Iterable[float], width: int, height: int, expand_ratio: float
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [float(value) for value in box_xyxy]
    box_width = x2 - x1
    box_height = y2 - y1
    x1 -= box_width * expand_ratio
    y1 -= box_height * expand_ratio
    x2 += box_width * expand_ratio
    y2 += box_height * expand_ratio
    clipped = (
        max(0, min(width - 1, int(round(x1)))),
        max(0, min(height - 1, int(round(y1)))),
        max(1, min(width, int(round(x2)))),
        max(1, min(height, int(round(y2)))),
    )
    left, top, right, bottom = clipped
    if right <= left:
        right = min(width, left + 1)
    if bottom <= top:
        bottom = min(height, top + 1)
    return left, top, right, bottom


def find_font(font_path: Path | None, size: int) -> Any | None:
    try:
        from PIL import ImageFont
    except ImportError:
        return None

    candidates = []
    if font_path is not None:
        candidates.append(font_path)
    candidates.extend(
        Path(path)
        for path in (
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.otf",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
            "/usr/share/fonts/truetype/arphic/uming.ttc",
            "/System/Library/Fonts/PingFang.ttc",
            "C:/Windows/Fonts/simhei.ttf",
        )
    )
    for candidate in candidates:
        if candidate.is_file():
            try:
                return ImageFont.truetype(str(candidate), size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def draw_annotations(
    image_bgr: Any,
    detections: list[PlateDetection],
    subset: str,
    font_path: Path | None,
) -> Any:
    cv2 = require_cv2()
    canvas = image_bgr.copy()
    if not detections:
        cv2.putText(
            canvas,
            f"{subset}: NO PLATE",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return canvas

    for detection_index, detection in enumerate(detections, start=1):
        x1, y1, x2, y2 = detection.box_xyxy
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)

    try:
        from PIL import Image, ImageDraw
        import numpy as np

        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        draw = ImageDraw.Draw(pil_image)
        image_width = canvas.shape[1]
        for detection_index, detection in enumerate(detections, start=1):
            x1, y1, x2, y2 = detection.box_xyxy
            label = (
                f"{detection_index}:{detection.pred or 'OCR_EMPTY'} "
                f"{detection.confidence:.2f}"
            )
            font_size = min(72, max(32, int((y2 - y1) * 0.55)))
            font = find_font(font_path, size=font_size)
            left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
            text_width = right - left
            text_height = bottom - top
            text_x = x1
            text_y = max(0, y1 - text_height - 8)
            while text_x + text_width + 16 > image_width and font_size > 24:
                font_size -= 4
                font = find_font(font_path, size=font_size)
                left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
                text_width = right - left
                text_height = bottom - top
                text_y = max(0, y1 - text_height - 8)
            draw.rectangle(
                [
                    text_x,
                    text_y,
                    text_x + text_width + 16,
                    text_y + text_height + 16,
                ],
                fill=(0, 160, 0),
            )
            draw.text(
                (text_x + 8, text_y + 8), label, font=font, fill=(255, 255, 255)
            )
        return cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2BGR)
    except Exception:
        for detection_index, detection in enumerate(detections, start=1):
            x1, y1, _x2, _y2 = detection.box_xyxy
            label = f"{detection_index}:{detection.pred or 'OCR_EMPTY'}"
            cv2.putText(
                canvas,
                label,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
        return canvas


def list_dataset_images(
    dataset_root: Path, selected_subsets: list[str], limit: int | None
) -> list[DatasetImage]:
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"测试集目录不存在: {dataset_root}")

    images: list[DatasetImage] = []
    for subset in selected_subsets:
        subset_dir = dataset_root / subset
        if not subset_dir.is_dir():
            raise FileNotFoundError(f"缺少子集目录: {subset_dir}")
        subset_images = sorted(
            path
            for path in subset_dir.iterdir()
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        )
        if limit is not None:
            subset_images = subset_images[:limit]
        images.extend(DatasetImage(path=path, subset=subset) for path in subset_images)
    if not images:
        raise FileNotFoundError(f"测试集中没有图片: {dataset_root}")
    return images


def load_truth_csv(path: Path | None, dataset_root: Path) -> dict[str, list[str]]:
    """Load optional GT labels.

    Accepted columns:
    * image / filename / path
    * plate / plates / gt / ground_truth

    Keys may be either image filename or relative path such as
    ``plates/example.jpg``.
    """
    if path is None:
        return {}
    if not path.is_file():
        raise FileNotFoundError(f"真值 CSV 不存在: {path}")

    truth: dict[str, list[str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"真值 CSV 没有表头: {path}")
        image_column = next(
            (
                name
                for name in ("image", "filename", "file", "path")
                if name in reader.fieldnames
            ),
            None,
        )
        plate_column = next(
            (
                name
                for name in ("plate", "plates", "gt", "ground_truth", "label")
                if name in reader.fieldnames
            ),
            None,
        )
        if image_column is None or plate_column is None:
            raise ValueError(
                "真值 CSV 需要包含 image/filename/path 和 "
                "plate/plates/gt/ground_truth 列"
            )
        for row in reader:
            image_value = row.get(image_column, "").strip().replace("\\", "/")
            plates = split_plate_text(row.get(plate_column, ""))
            if not image_value or not plates:
                continue
            truth[image_value] = plates
            truth[Path(image_value).name] = plates
            try:
                relative = Path(image_value)
                if relative.is_absolute():
                    relative = relative.relative_to(dataset_root)
                truth[relative.as_posix()] = plates
            except ValueError:
                pass
    return truth


def ground_truth_for(
    image_path: Path, subset: str, dataset_root: Path, truth: dict[str, list[str]]
) -> list[str]:
    relative = image_path.relative_to(dataset_root).as_posix()
    return (
        truth.get(relative)
        or truth.get(f"{subset}/{image_path.name}")
        or truth.get(image_path.name)
        or decode_ccpd_plate_from_filename(image_path)
    )


def result_metrics(results: list[ImageResult]) -> dict[str, Any]:
    total = len(results)
    detected_images = sum(bool(result.detections) for result in results)
    gt_results = [result for result in results if result.gt_plates]
    total_gt_chars = sum(len(plate) for result in gt_results for plate in result.gt_plates)
    total_distance = sum(result.char_distance or 0 for result in gt_results)
    full_matches = sum(result.full_match is True for result in gt_results)
    return {
        "total_images": total,
        "detected_images": detected_images,
        "detection_rate": detected_images / total if total else 0.0,
        "gt_images": len(gt_results),
        "full_matches": full_matches,
        "full_match_accuracy": (
            full_matches / len(gt_results) if gt_results else None
        ),
        "character_level_accuracy": (
            max(0.0, 1.0 - total_distance / total_gt_chars)
            if total_gt_chars
            else None
        ),
        "total_edit_distance": total_distance if gt_results else None,
        "total_gt_characters": total_gt_chars if gt_results else 0,
        "total_detections": sum(len(result.detections) for result in results),
    }


def write_predictions_csv(results: list[ImageResult], output_dir: Path) -> None:
    with (output_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "subset",
                "image",
                "ground_truth",
                "predictions",
                "detected_count",
                "confidences",
                "boxes_xyxy",
                "full_match",
                "char_edit_distance",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "subset": result.subset,
                    "image": result.image.name,
                    "ground_truth": "|".join(result.gt_plates),
                    "predictions": "|".join(
                        detection.pred for detection in result.detections
                    ),
                    "detected_count": len(result.detections),
                    "confidences": "|".join(
                        f"{detection.confidence:.6f}"
                        for detection in result.detections
                    ),
                    "boxes_xyxy": json.dumps(
                        [detection.box_xyxy for detection in result.detections],
                        ensure_ascii=False,
                    ),
                    "full_match": "" if result.full_match is None else result.full_match,
                    "char_edit_distance": (
                        "" if result.char_distance is None else result.char_distance
                    ),
                }
            )


def write_metrics(results: list[ImageResult], output_dir: Path) -> dict[str, Any]:
    by_subset = {
        subset: result_metrics(
            [result for result in results if result.subset == subset]
        )
        for subset in SUBSETS
    }
    metrics = {
        "overall": result_metrics(results),
        "by_subset": by_subset,
        "note": (
            "Full Match Accuracy 和 Character-level Accuracy 只有在存在真值车牌号"
            "时计算；当前 university_plates 默认文件名不含真值。"
        ),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return metrics


def format_accuracy(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.4f}"


def print_subset_summary(metrics: dict[str, Any]) -> None:
    print("\n各子集检测效果")
    print("subset,total,detected,detections,det_rate,full_match_acc,char_acc")
    for subset, item in metrics["by_subset"].items():
        print(
            f"{subset},"
            f"{item['total_images']},"
            f"{item['detected_images']},"
            f"{item['total_detections']},"
            f"{item['detection_rate']:.4f},"
            f"{format_accuracy(item['full_match_accuracy'])},"
            f"{format_accuracy(item['character_level_accuracy'])}"
        )
    overall = metrics["overall"]
    print(
        "overall,"
        f"{overall['total_images']},"
        f"{overall['detected_images']},"
        f"{overall['total_detections']},"
        f"{overall['detection_rate']:.4f},"
        f"{format_accuracy(overall['full_match_accuracy'])},"
        f"{format_accuracy(overall['character_level_accuracy'])}"
    )


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    if not args.weights.is_file():
        raise FileNotFoundError(f"YOLO 权重不存在: {args.weights}")

    cv2 = require_cv2()
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError("缺少 ultralytics，请先执行 `uv sync` 安装依赖") from error

    dataset_images = list_dataset_images(args.dataset_root, args.subsets, args.limit)
    truth = load_truth_csv(args.truth_csv, args.dataset_root)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    annotated_root = args.output_dir / "annotated"
    crops_root = args.output_dir / "crops"
    if args.save_annotated:
        annotated_root.mkdir(parents=True, exist_ok=True)
    if args.save_crops:
        crops_root.mkdir(parents=True, exist_ok=True)

    yolo = YOLO(str(args.weights))
    ocr = build_paddle_recognizer(args.ocr_device)

    results: list[ImageResult] = []
    source = [str(item.path) for item in dataset_images]
    result_stream = yolo.predict(
        source=source,
        stream=True,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        batch=args.batch,
        max_det=args.max_det,
        verbose=False,
    )
    subset_by_path = {item.path.resolve(): item.subset for item in dataset_images}

    for index, result in enumerate(result_stream, start=1):
        image_path = Path(result.path)
        subset = subset_by_path[image_path.resolve()]
        gt_plates = ground_truth_for(image_path, subset, args.dataset_root, truth)
        image = result.orig_img
        height, width = image.shape[:2]
        detections: list[PlateDetection] = []

        if result.boxes is not None and len(result.boxes) > 0:
            boxes = sorted(
                result.boxes,
                key=lambda box: float(box.conf[0].item()),
                reverse=True,
            )
            for detection_index, box_result in enumerate(boxes, start=1):
                confidence = float(box_result.conf[0].item())
                box = clip_box(
                    box_result.xyxy[0].tolist(), width, height, args.crop_expand
                )
                x1, y1, x2, y2 = box
                crop = image[y1:y2, x1:x2]
                pred = recognize_crop(ocr, crop)
                detections.append(
                    PlateDetection(pred=pred, confidence=confidence, box_xyxy=box)
                )
                if args.save_crops:
                    crop_dir = crops_root / subset
                    crop_dir.mkdir(parents=True, exist_ok=True)
                    crop_name = f"{image_path.stem}_det{detection_index}{image_path.suffix}"
                    cv2.imwrite(str(crop_dir / crop_name), crop)

        pred_plates = [detection.pred for detection in detections if detection.pred]
        if gt_plates:
            distance = plate_list_distance(gt_plates, pred_plates)
            is_full_match = full_match(gt_plates, pred_plates)
        else:
            distance = None
            is_full_match = None

        image_result = ImageResult(
            image=image_path,
            subset=subset,
            gt_plates=gt_plates,
            detections=detections,
            char_distance=distance,
            full_match=is_full_match,
        )
        results.append(image_result)

        if args.save_annotated:
            annotated_dir = annotated_root / subset
            annotated_dir.mkdir(parents=True, exist_ok=True)
            annotated = draw_annotations(image, detections, subset, args.font)
            cv2.imwrite(str(annotated_dir / image_path.name), annotated)

        gt_text = "|".join(gt_plates) if gt_plates else "N/A"
        pred_text = "|".join(pred_plates) if pred_plates else "<EMPTY>"
        print(
            f"[{index:>3}/{len(dataset_images)}] {subset}/{image_path.name} "
            f"DET={len(detections)} GT={gt_text} PRED={pred_text}"
        )

    write_predictions_csv(results, args.output_dir)
    metrics = write_metrics(results, args.output_dir)
    print_subset_summary(metrics)
    print(
        "\n关键输出\n"
        f"逐图结果: {args.output_dir / 'predictions.csv'}\n"
        f"指标文件: {args.output_dir / 'metrics.json'}"
    )
    if args.save_annotated:
        print(f"分类标注图: {annotated_root}")
    if args.save_crops:
        print(f"分类裁剪图: {crops_root}")
    if metrics["overall"]["gt_images"] == 0:
        print(
            "未发现真值车牌号，因此 Full Match Accuracy 和 "
            "Character-level Accuracy 输出为 N/A；如需计算准确率，"
            "请使用 --truth-csv 提供 image,plate 或 image,plates 列。"
        )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate university_plates with YOLO detection + PaddleOCR OCR."
    )
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--subsets",
        nargs="+",
        default=list(SUBSETS),
        choices=SUBSETS,
        help="要评估的 university_plates 子集",
    )
    parser.add_argument(
        "--truth-csv",
        type=Path,
        default=None,
        help="可选真值 CSV，列名支持 image/filename/path 和 plate/plates/gt",
    )
    parser.add_argument("--limit", type=int, default=None, help="每个子集只评估前 N 张")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--max-det", type=int, default=10, help="每张图最多检测几个车牌")
    parser.add_argument("--device", default=None, help="YOLO 推理设备，例如 0 或 cpu")
    parser.add_argument("--ocr-device", default=None, help="PaddleOCR 设备，例如 gpu:0 或 cpu")
    parser.add_argument("--crop-expand", type=float, default=0.05, help="裁剪框外扩比例")
    parser.add_argument(
        "--font",
        type=Path,
        default=DEFAULT_FONT,
        help="绘制中文车牌号的字体路径",
    )
    parser.add_argument("--save-crops", action="store_true", help="保存车牌裁剪图")
    parser.add_argument(
        "--no-save-annotated",
        dest="save_annotated",
        action="store_false",
        help="不保存带车牌框和号码的标注图片",
    )
    parser.set_defaults(save_annotated=True)
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit 必须为正整数")
    if args.max_det <= 0:
        raise ValueError("--max-det 必须为正整数")
    if args.crop_expand < 0:
        raise ValueError("--crop-expand 不能为负数")
    return args


if __name__ == "__main__":
    evaluate(parse_args())
