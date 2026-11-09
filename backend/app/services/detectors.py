from pathlib import Path
from app.services.tracking import Detection


class DetectorUnavailable(RuntimeError):
    pass


def load_numberplate_detector_weights() -> str:
    """Return the path to the pretrained numberplate detector weights.

    Raises:
        DetectorUnavailable: If the pretrained weights are not found.
    """
    from huggingface_hub import hf_hub_download

    weights_path = hf_hub_download(
        repo_id="ishaqinu/yolo26n-license-plate-detection",
        filename="alpr_26n.pt"
    )
    if not weights_path.is_file():
        raise DetectorUnavailable(
            f"Numberplate detector weights not found: {weights_path}")

    return str(weights_path)


class UltralyticsDetector:
    """Optional YOLO adapter. Model weights are supplied by the operator, not bundled."""

    def __init__(self, weights: str, class_ids: set[int] | None = None, confidence: float = 0.25):
        if not weights or not Path(weights).is_file():
            raise DetectorUnavailable(
                f"Model weights not found: {weights or '(not configured)'}")

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise DetectorUnavailable(
                "Install the optional 'ultralytics' package to run YOLO inference") from exc

        self.model = YOLO(weights)
        self.class_ids = class_ids
        self.confidence = confidence

    def detect(self, frame) -> list[Detection]:
        results = self.model.predict(
            frame, conf=self.confidence, verbose=False)
        detections: list[Detection] = []

        if not results:
            return detections

        for box in results[0].boxes:
            cls_id = int(box.cls.item())
            if self.class_ids is not None and cls_id not in self.class_ids:
                continue

            coords = [float(v) for v in box.xyxy[0].tolist()]
            detections.append(
                Detection(coords, float(box.conf.item()), cls_id))

        return detections


class HybridPlateOCR:
    """Pretrained OCR ensemble. It never trains or modifies either OCR model."""

    def __init__(self):
        self.paddle = None
        self.easy = None
        errors = []

        try:
            from paddleocr import PaddleOCR

            try:
                self.paddle = PaddleOCR(
                    use_angle_cls=True, lang="en", show_log=False)
            except TypeError:
                self.paddle = PaddleOCR(use_angle_cls=True, lang="en")
        except Exception as exc:
            errors.append(f"PaddleOCR: {exc}")

        try:
            import easyocr
            self.easy = easyocr.Reader(["en"], gpu=False, verbose=False)
        except Exception as exc:
            errors.append(f"EasyOCR: {exc}")

        if self.paddle is None and self.easy is None:
            raise DetectorUnavailable(
                "No OCR engine is available. Install paddleocr or easyocr. " + "; ".join(errors))

        self.available_engines = [name for name, engine in (
            ("paddleocr", self.paddle), ("easyocr", self.easy)) if engine is not None]

    @staticmethod
    def _normalise(text):
        import re
        return re.sub(r"[^A-Z0-9]", "", str(text).upper())

    def _read_paddle(self, image):
        try:
            result = self.paddle.ocr(image, cls=True)
        except TypeError:
            result = self.paddle.ocr(image)

        found = []
        for line_group in result or []:
            for item in line_group or []:
                if len(item) >= 2 and isinstance(item[1], (tuple, list)):
                    text, confidence = item[1][0], float(item[1][1])
                    if text:
                        found.append(
                            (str(text), max(0.0, min(1.0, confidence))))

        return found

    def _read_easy(self, image):
        result = self.easy.readtext(image, detail=1, paragraph=False)
        return [(str(item[1]), max(0.0, min(1.0, float(item[2]))))
                for item in (result or []) if len(item) >= 3 and item[1]]

    def recognize(self, image) -> list[dict]:
        """Try raw-enhanced, grayscale and adaptive views; return one conservative crop-level read."""
        from app.services.plates.image_quality import ocr_preprocessing_variants

        variants = ocr_preprocessing_variants(image)
        candidates = []

        for variant_name, variant in variants.items():
            for engine_name, engine in (("paddleocr", self.paddle), ("easyocr", self.easy)):
                if engine is None:
                    continue

                try:
                    outputs = self._read_paddle(
                        variant) if engine_name == "paddleocr" else self._read_easy(variant)
                except Exception:
                    continue

                for text, confidence in outputs:
                    normalized = self._normalise(text)
                    if normalized:
                        candidates.append({"text": normalized, "confidence": confidence,
                                           "engine": engine_name, "variant": variant_name})

        if not candidates:
            return []

        # Prefer agreement across different pretrained engines. Repeated variants from
        # one engine are useful evidence, but are not treated as independent engines.
        from collections import defaultdict

        grouped = defaultdict(list)
        for candidate in candidates:
            grouped[candidate["text"]].append(candidate)

        ranked = sorted(grouped.items(), key=lambda pair: (
            len({item["engine"] for item in pair[1]}), len(pair[1]),
            sum(item["confidence"] for item in pair[1]) / len(pair[1])), reverse=True)
        text, evidence = ranked[0]
        engines = {item["engine"] for item in evidence}
        variants_used = {item["variant"] for item in evidence}
        mean_confidence = sum(item["confidence"]
                              for item in evidence) / len(evidence)
        confidence = mean_confidence

        if len(engines) == 1 and len(variants_used) < 2:
            confidence = min(confidence, 0.55)

        if len(ranked) > 1:
            second = set(item["engine"] for item in ranked[1][1])
            if engines == second and len(evidence) == len(ranked[1][1]):
                confidence = min(confidence, 0.55)

        return [{"text": text, "confidence": round(confidence, 4),
                 "engines": sorted(engines), "variants": sorted(variants_used),
                 "candidate_count": len(candidates)}]


# Backwards-compatible name for callers that used the earlier single-engine adapter.
PaddlePlateOCR = HybridPlateOCR
