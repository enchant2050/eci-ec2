"""Unit tests for OCR engine."""

from src.ocr_engine import OCREngine


def test_extract_text_uses_tesseract_conf_key(monkeypatch):
    """Tesseract image_to_data exposes confidence as conf, not confidence."""

    def fake_image_to_data(*args, **kwargs):
        return {"conf": ["-1", "91.5", "84"]}

    def fake_image_to_string(*args, **kwargs):
        return "Name: TEST VOTER"

    monkeypatch.setattr("src.ocr_engine.pytesseract.image_to_data", fake_image_to_data)
    monkeypatch.setattr("src.ocr_engine.pytesseract.image_to_string", fake_image_to_string)

    text, confidence = OCREngine.extract_text("card.jpg")

    assert text == "Name: TEST VOTER"
    assert confidence == 87.75
