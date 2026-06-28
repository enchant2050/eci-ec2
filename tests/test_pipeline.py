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


def test_process_pdf_defaults_to_voter_pages(monkeypatch, tmp_path):
    processed_pages = []
    conversion_args = {}

    def fake_convert_pdf_to_images(pdf_path, output_dir, dpi, first_page=None, last_page=None):
        conversion_args["first_page"] = first_page
        conversion_args["last_page"] = last_page
        return [f"{output_dir}/page_{page:04d}.jpg" for page in range(first_page, last_page + 1)]

    def fake_process_page(self, image_path, page_number, pdf_name, work_dir):
        processed_pages.append(page_number)
        return {
            "page_number": page_number,
            "record_count": 0,
            "records": [],
            "processing_time": 0,
        }

    monkeypatch.setattr("src.pipeline.ImageProcessor.get_pdf_page_count", lambda pdf_path: 10)
    monkeypatch.setattr("src.pipeline.ImageProcessor.convert_pdf_to_images", fake_convert_pdf_to_images)
    monkeypatch.setattr("src.pipeline.OCRPipeline._process_page", fake_process_page)

    result = OCRPipeline().process_pdf(str(tmp_path / "sample.pdf"))

    assert processed_pages == [3, 4, 5, 6, 7, 8, 9]
    assert conversion_args == {"first_page": 3, "last_page": 9}
    assert result["start_page"] == 3
    assert result["end_page"] == 9


def test_process_pdf_converts_only_explicit_page_range(monkeypatch, tmp_path):
    processed_pages = []
    conversion_args = {}

    def fake_convert_pdf_to_images(pdf_path, output_dir, dpi, first_page=None, last_page=None):
        conversion_args["first_page"] = first_page
        conversion_args["last_page"] = last_page
        return [f"{output_dir}/page_{page:04d}.jpg" for page in range(first_page, last_page + 1)]

    def fake_process_page(self, image_path, page_number, pdf_name, work_dir):
        processed_pages.append(page_number)
        return {
            "page_number": page_number,
            "record_count": 0,
            "records": [],
            "processing_time": 0,
        }

    monkeypatch.setattr("src.pipeline.ImageProcessor.get_pdf_page_count", lambda pdf_path: 10)
    monkeypatch.setattr("src.pipeline.ImageProcessor.convert_pdf_to_images", fake_convert_pdf_to_images)
    monkeypatch.setattr("src.pipeline.OCRPipeline._process_page", fake_process_page)

    result = OCRPipeline().process_pdf(
        str(tmp_path / "sample.pdf"),
        start_page=3,
        end_page=3,
    )

    assert processed_pages == [3]
    assert conversion_args == {"first_page": 3, "last_page": 3}
    assert result["start_page"] == 3
    assert result["end_page"] == 3
