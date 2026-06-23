"""Detect and recognize CCPD2019 license plates with YOLO + PaddleOCR.

The script does not train anything.  It loads the trained YOLO detector from
``weights/yolo26n_ccpd_best.pt``, detects the plate in each image from
``CCPD2019/yolo_ccpd_base/images/test``, crops the detected plate region, runs
PaddleOCR on the crop, then writes:

* annotated images with "plate box + plate number";
* per-image predictions in CSV format;
* Full Match Accuracy and Character-level Accuracy metrics.
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
DEFAULT_IMAGES = ROOT / "CCPD2019" / "yolo_ccpd_base" / "images" / "test"
DEFAULT_OUTPUT = ROOT / "runs" / "plate_ocr_eval"

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


@dataclass
class Prediction:
    image: Path
    gt: str
    pred: str
    detected: bool
    confidence: float
    box_xyxy: tuple[int, int, int, int] | None
    char_distance: int


def require_cv2() -> Any:
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("缺少 opencv-python，请先执行 `uv sync` 安装依赖") from error
    return cv2


def decode_plate_from_filename(path: Path) -> str:
    """Decode the 7-character CCPD plate number from the image filename."""
    try:
        encoded = path.stem.split("-")[4]
        indexes = [int(value) for value in encoded.split("_")]
    except (IndexError, ValueError) as error:
        raise ValueError(f"无法从文件名解析车牌编码: {path.name}") from error

    if len(indexes) != 7:
        raise ValueError(f"车牌编码不是 7 位: {path.name}")
    try:
        return (
            PROVINCES[indexes[0]]
            + ALPHABETS[indexes[1]]
            + "".join(ADS[index] for index in indexes[2:])
        )
    except IndexError as error:
        raise ValueError(f"车牌编码索引越界: {path.name}") from error


def normalize_plate(text: str) -> str:
    """Normalize OCR text before comparing it with CCPD ground truth."""
    text = unicodedata.normalize("NFKC", text).upper()
    # Keep Chinese characters and ASCII letters/digits.  Drop OCR separators,
    # whitespace, punctuation, confidence artifacts, etc.
    return "".join(re.findall(r"[\u4e00-\u9fffA-Z0-9]", text))


def edit_distance(left: str, right: str) -> int:
    """Levenshtein edit distance for character-level OCR accuracy."""
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


def list_images(images_dir: Path, limit: int | None) -> list[Path]:
    if not images_dir.is_dir():
        raise FileNotFoundError(f"测试集目录不存在: {images_dir}")
    images = sorted(
        path
        for path in images_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )
    if not images:
        raise FileNotFoundError(f"测试集目录没有图片: {images_dir}")
    return images[:limit] if limit is not None else images


def accepted_kwargs(function: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return kwargs
    if any(param.kind == param.VAR_KEYWORD for param in signature.parameters.values()):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in signature.parameters}


def build_paddle_recognizer(lang: str, device: str | None) -> Any:
    """Create a PaddleOCR recognizer for already-cropped plate images.

    YOLO has already localized and cropped the plate.  Using PaddleOCR's
    TextRecognition avoids running a second text detector on the crop, and also
    bypasses the Paddle/PaddleX CPU oneDNN text-detection path that can raise:
    ``ConvertPirAttribute2RuntimeAttribute not support ...``.
    """
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
        # Fall back to the full OCR pipeline for older PaddleOCR installations.
        pass

    new_kwargs: dict[str, Any] = {
        "lang": lang,
        "device": device,
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "use_textline_orientation": True,
    }
    old_kwargs: dict[str, Any] = {
        "lang": lang,
        "use_angle_cls": True,
        "show_log": False,
    }
    if device is not None:
        old_kwargs["use_gpu"] = not device.lower().startswith("cpu")

    errors: list[Exception] = []
    for kwargs in (new_kwargs, old_kwargs):
        clean_kwargs = {
            key: value
            for key, value in accepted_kwargs(PaddleOCR.__init__, kwargs).items()
            if value is not None
        }
        try:
            return PaddleOCR(**clean_kwargs)
        except Exception as error:  # PaddleOCR raises different errors by version.
            errors.append(error)

    message = "; ".join(str(error) for error in errors[-2:])
    raise RuntimeError(f"初始化 PaddleOCR 识别器失败: {message}")


def extract_texts_from_ocr_result(result: Any) -> list[tuple[str, float]]:
    """Extract recognized texts from multiple PaddleOCR result formats."""
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
            # Fall through to the legacy API when available.
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
    # A cropped plate should normally contain one text line.  If OCR splits it,
    # concatenate all detected fragments in result order.
    joined = "".join(text for text, _score in candidates)
    return normalize_plate(joined)


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
    return (
        max(0, min(width - 1, int(round(x1)))),
        max(0, min(height - 1, int(round(y1)))),
        max(1, min(width, int(round(x2)))),
        max(1, min(height, int(round(y2)))),
    )


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


def draw_annotation(
    image_bgr: Any,
    box: tuple[int, int, int, int] | None,
    text: str,
    font_path: Path | None,
) -> Any:
    cv2 = require_cv2()
    canvas = image_bgr.copy()
    if box is None:
        cv2.putText(
            canvas,
            "NO PLATE",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return canvas

    x1, y1, x2, y2 = box
    cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)

    label = text or "OCR_EMPTY"
    try:
        from PIL import Image, ImageDraw
        import numpy as np

        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        draw = ImageDraw.Draw(pil_image)
        font = find_font(font_path, size=max(20, int((y2 - y1) * 0.45)))
        left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
        text_width = right - left
        text_height = bottom - top
        text_x = x1
        text_y = max(0, y1 - text_height - 8)
        draw.rectangle(
            [text_x, text_y, text_x + text_width + 8, text_y + text_height + 8],
            fill=(0, 160, 0),
        )
        draw.text((text_x + 4, text_y + 4), label, font=font, fill=(255, 255, 255))
        return cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2BGR)
    except Exception:
        cv2.putText(
            canvas,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        return canvas


def write_metrics(predictions: list[Prediction], output_dir: Path) -> dict[str, Any]:
    total = len(predictions)
    full_matches = sum(item.gt == item.pred for item in predictions)
    detected = sum(item.detected for item in predictions)
    total_gt_chars = sum(len(item.gt) for item in predictions)
    total_distance = sum(item.char_distance for item in predictions)
    char_accuracy = (
        max(0.0, 1.0 - total_distance / total_gt_chars) if total_gt_chars else 0.0
    )
    metrics = {
        "total_images": total,
        "detected_images": detected,
        "detection_rate": detected / total if total else 0.0,
        "full_match_accuracy": full_matches / total if total else 0.0,
        "character_level_accuracy": char_accuracy,
        "full_matches": full_matches,
        "total_edit_distance": total_distance,
        "total_gt_characters": total_gt_chars,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return metrics


def write_predictions_csv(predictions: list[Prediction], output_dir: Path) -> None:
    with (output_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image",
                "ground_truth",
                "prediction",
                "full_match",
                "char_edit_distance",
                "detected",
                "confidence",
                "box_xyxy",
            ],
        )
        writer.writeheader()
        for item in predictions:
            writer.writerow(
                {
                    "image": item.image.name,
                    "ground_truth": item.gt,
                    "prediction": item.pred,
                    "full_match": item.gt == item.pred,
                    "char_edit_distance": item.char_distance,
                    "detected": item.detected,
                    "confidence": f"{item.confidence:.6f}",
                    "box_xyxy": "" if item.box_xyxy is None else list(item.box_xyxy),
                }
            )


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    if not args.weights.is_file():
        raise FileNotFoundError(f"YOLO 权重不存在: {args.weights}")

    cv2 = require_cv2()
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError("缺少 ultralytics，请先执行 `uv sync` 安装依赖") from error

    images = list_images(args.images_dir, args.limit)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    annotated_dir = args.output_dir / "annotated"
    crops_dir = args.output_dir / "crops"
    if args.save_annotated:
        annotated_dir.mkdir(parents=True, exist_ok=True)
    if args.save_crops:
        crops_dir.mkdir(parents=True, exist_ok=True)

    yolo = YOLO(str(args.weights))
    ocr = build_paddle_recognizer(args.ocr_lang, args.ocr_device)

    predictions: list[Prediction] = []
    source = [str(path) for path in images]
    results = yolo.predict(
        source=source,
        stream=True,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        batch=args.batch,
        max_det=1,
        verbose=False,
    )

    for index, result in enumerate(results, start=1):
        image_path = Path(result.path)
        gt = decode_plate_from_filename(image_path)
        image = result.orig_img
        height, width = image.shape[:2]
        box: tuple[int, int, int, int] | None = None
        confidence = 0.0
        pred = ""

        if result.boxes is not None and len(result.boxes) > 0:
            best = result.boxes[0]
            confidence = float(best.conf[0].item())
            box = clip_box(best.xyxy[0].tolist(), width, height, args.crop_expand)
            x1, y1, x2, y2 = box
            crop = image[y1:y2, x1:x2]
            if args.save_crops:
                cv2.imwrite(str(crops_dir / image_path.name), crop)
            pred = recognize_crop(ocr, crop)

        pred = normalize_plate(pred)
        distance = edit_distance(gt, pred)
        item = Prediction(
            image=image_path,
            gt=gt,
            pred=pred,
            detected=box is not None,
            confidence=confidence,
            box_xyxy=box,
            char_distance=distance,
        )
        predictions.append(item)

        if args.save_annotated:
            annotated = draw_annotation(image, box, pred, args.font)
            cv2.imwrite(str(annotated_dir / image_path.name), annotated)

        print(
            f"[{index:>5}/{len(images)}] {image_path.name} "
            f"GT={gt} PRED={pred or '<EMPTY>'} "
            f"CONF={confidence:.3f} MATCH={gt == pred}"
        )

    write_predictions_csv(predictions, args.output_dir)
    metrics = write_metrics(predictions, args.output_dir)
    print(
        "\n评估完成\n"
        f"图片数: {metrics['total_images']}\n"
        f"检测率: {metrics['detection_rate']:.4f}\n"
        f"整牌识别准确率 Full Match Accuracy: "
        f"{metrics['full_match_accuracy']:.4f}\n"
        f"字符级识别准确率 Character-level Accuracy: "
        f"{metrics['character_level_accuracy']:.4f}\n"
        f"逐图结果: {args.output_dir / 'predictions.csv'}\n"
        f"指标文件: {args.output_dir / 'metrics.json'}"
    )
    if args.save_annotated:
        print(f"标注图片: {annotated_dir}")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use YOLO plate detection + PaddleOCR to evaluate CCPD test OCR."
    )
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=None, help="只评估前 N 张，默认全量")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None, help="YOLO 推理设备，例如 0 或 cpu")
    parser.add_argument("--ocr-device", default=None, help="PaddleOCR 设备，例如 gpu:0 或 cpu")
    parser.add_argument("--ocr-lang", default="ch")
    parser.add_argument("--crop-expand", type=float, default=0.05, help="裁剪框外扩比例")
    parser.add_argument("--font", type=Path, default=None, help="绘制中文车牌号的字体路径")
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
    if args.crop_expand < 0:
        raise ValueError("--crop-expand 不能为负数")
    return args


if __name__ == "__main__":
    evaluate(parse_args())
