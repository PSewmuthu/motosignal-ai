import cv2


def crop_box(frame, bbox):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return None

    return frame[y1:y2, x1:x2].copy()


def quality_metrics(crop) -> dict:
    if crop is None or crop.size == 0:
        return {"width": 0, "height": 0, "laplacian_variance": 0.0}

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return {"width": int(crop.shape[1]), "height": int(crop.shape[0]),
            "laplacian_variance": float(cv2.Laplacian(gray, cv2.CV_64F).var())}


def preprocess_plate(crop):
    """Conservative colour enhancement; avoid aggressive filters that erase character strokes."""
    if crop is None or crop.size == 0:
        return None

    h, w = crop.shape[:2]
    scale = max(1.0, 160.0 / max(h, 1))
    resized = cv2.resize(crop, (max(1, int(w * scale)),
                         max(1, int(h * scale))), interpolation=cv2.INTER_CUBIC)
    lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = cv2.merge((clahe.apply(l), a, b))

    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


def ocr_preprocessing_variants(crop) -> dict:
    """Return a small set of alternative views for OCR; no learned model is changed."""
    enhanced = preprocess_plate(crop)
    if enhanced is None:
        return {}

    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    # Mild denoising preserves edges better than strong smoothing on small characters.
    denoised = cv2.bilateralFilter(gray, 5, 35, 35)
    adaptive = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 31, 11)

    return {"enhanced": enhanced, "gray": gray, "adaptive": adaptive}
