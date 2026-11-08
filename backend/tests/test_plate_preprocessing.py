import numpy as np

from app.services.plates.image_quality import ocr_preprocessing_variants, preprocess_plate, quality_metrics


def test_preprocessing_returns_multiple_valid_variants():
    # Synthetic high-contrast crop is used only to test array handling, not OCR accuracy.
    crop = np.full((48, 160, 3), 245, dtype=np.uint8)
    crop[12:36, 20:30] = 15
    variants = ocr_preprocessing_variants(crop)

    assert set(variants) == {"enhanced", "gray", "adaptive"}
    assert all(value.size > 0 for value in variants.values())
    assert variants["enhanced"].ndim == 3
    assert variants["gray"].ndim == 2
    assert variants["adaptive"].ndim == 2


def test_empty_crop_is_handled_safely():
    assert preprocess_plate(None) is None
    assert ocr_preprocessing_variants(None) == {}
    assert quality_metrics(None)["width"] == 0
