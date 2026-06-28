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


def test_text_from_data_reconstructs_lines():
    data = {
        "text": ["Name:", "TEST", "VOTER", "", "Age:", "45"],
        "block_num": [1, 1, 1, 1, 1, 1],
        "par_num": [1, 1, 1, 1, 1, 1],
        "line_num": [1, 1, 1, 1, 2, 2],
        "word_num": [1, 2, 3, 4, 1, 2],
    }

    assert OCREngine._text_from_data(data) == "Name: TEST VOTER\nAge: 45"
