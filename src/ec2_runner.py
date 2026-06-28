"""Command line runner for EC2/manual OCR processing."""
import argparse
import json
import os
from pathlib import Path
from typing import Optional

from src.pipeline import OCRPipeline
from src.logger import logger as json_logger


def process_pdf(
    pdf_path: str,
    output_path: str,
    work_dir: str,
    db_connection: Optional[str],
    skip_db_insert: bool,
    max_workers: int,
    start_page: Optional[int] = None,
    end_page: Optional[int] = None,
    all_pages: bool = False,
    exhaustive_ocr: bool = False,
    ocr_timeout: int = 15,
) -> dict:
    """Run the OCR pipeline for a single PDF."""
    if not skip_db_insert and not db_connection:
        json_logger.warning("DATABASE_URL is not set; processing JSON only and skipping database insert")

    pipeline = OCRPipeline(
        db_connection_string=None if skip_db_insert else db_connection,
        max_workers=max_workers,
        exhaustive_ocr=exhaustive_ocr,
        ocr_timeout=ocr_timeout,
    )
    result = pipeline.process_pdf(
        pdf_path,
        work_dir=work_dir,
        skip_db_insert=skip_db_insert or not db_connection,
        start_page=start_page,
        end_page=end_page,
        all_pages=all_pages,
    )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    json_logger.info("Results written", output_path=str(output))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Process an electoral roll PDF on EC2")
    parser.add_argument("pdf", help="Path to the PDF to process")
    parser.add_argument(
        "--output",
        default="outputs/result.json",
        help="Path for the JSON output file",
    )
    parser.add_argument(
        "--work-dir",
        default="/tmp/electoral_roll_ocr",
        help="Temporary working directory for page/card images",
    )
    parser.add_argument(
        "--db-url",
        default=os.environ.get("DATABASE_URL"),
        help="PostgreSQL URL. Defaults to DATABASE_URL.",
    )
    parser.add_argument(
        "--skip-db",
        action="store_true",
        help="Process and write JSON without inserting records into the database",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=int(os.environ.get("OCR_MAX_WORKERS", "4")),
        help="Number of parallel OCR workers",
    )
    parser.add_argument(
        "--start-page",
        type=int,
        default=None,
        help="First 1-indexed PDF page to process. Defaults to page 3 for normal electoral rolls.",
    )
    parser.add_argument(
        "--end-page",
        type=int,
        default=None,
        help="Last 1-indexed PDF page to process. Defaults to the page before the summary page.",
    )
    parser.add_argument(
        "--all-pages",
        action="store_true",
        help="Process every PDF page, including cover/map/summary pages.",
    )
    parser.add_argument(
        "--exhaustive-ocr",
        action="store_true",
        default=os.environ.get("OCR_EXHAUSTIVE", "0").lower() in {"1", "true", "yes"},
        help="Run slower multi-pass OCR for harder scans.",
    )
    parser.add_argument(
        "--ocr-timeout",
        type=int,
        default=int(os.environ.get("OCR_TIMEOUT_SECONDS", "15")),
        help="Maximum seconds for each Tesseract call.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    process_pdf(
        pdf_path=args.pdf,
        output_path=args.output,
        work_dir=args.work_dir,
        db_connection=args.db_url,
        skip_db_insert=args.skip_db,
        max_workers=args.max_workers,
        start_page=args.start_page,
        end_page=args.end_page,
        all_pages=args.all_pages,
        exhaustive_ocr=args.exhaustive_ocr,
        ocr_timeout=args.ocr_timeout,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
