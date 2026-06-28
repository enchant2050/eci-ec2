"""Unit tests for OCR pipeline post-processing."""

from src.pipeline import OCRPipeline


def test_fill_missing_serial_numbers_from_card_sequence():
    pages = [
        {
            "records": [
                {"page_number": 7, "card_number": 1, "serial_number": None},
                {"page_number": 7, "card_number": 2, "serial_number": None},
                {"page_number": 7, "card_number": 9, "serial_number": 129},
            ]
        }
    ]

    OCRPipeline._fill_missing_serial_numbers(pages)

    assert pages[0]["records"][0]["serial_number"] == 121
    assert pages[0]["records"][1]["serial_number"] == 122
    assert pages[0]["records"][2]["serial_number"] == 129
