"""Main OCR Pipeline Orchestration"""
import os
import json
from pathlib import Path
from typing import List, Dict, Optional
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
import time

from src.image_processor import ImageProcessor
from src.card_detector import CardDetector
from src.ocr_engine import OCREngine
from src.nlp_parser import NLPParser, VoterRecord
from src.database import DatabaseManager, ElectoralRollRecord
from src.logger import logger as json_logger

logger = logging.getLogger(__name__)


class OCRPipeline:
    """Complete OCR processing pipeline"""
    
    def __init__(
        self,
        db_connection_string: Optional[str] = None,
        max_workers: int = 4,
        exhaustive_ocr: bool = False,
        ocr_timeout: int = 15,
    ):
        """
        Initialize pipeline
        
        Args:
            db_connection_string: PostgreSQL connection string
            max_workers: Number of parallel workers for card OCR
            exhaustive_ocr: Run slower multi-pass OCR for each card
            ocr_timeout: Max seconds per Tesseract call
        """
        self.db_manager = None
        if db_connection_string:
            self.db_manager = DatabaseManager(db_connection_string)
            self.db_manager.create_tables()
        
        self.max_workers = max_workers
        self.exhaustive_ocr = exhaustive_ocr
        self.ocr_timeout = ocr_timeout
    
    def process_pdf(
        self,
        pdf_path: str,
        work_dir: str = "/tmp/electoral_roll_ocr",
        skip_db_insert: bool = False,
        start_page: Optional[int] = None,
        end_page: Optional[int] = None,
        all_pages: bool = False,
    ) -> Dict:
        """
        Process entire PDF document
        
        Args:
            pdf_path: Path to input PDF
            work_dir: Working directory for temporary files
            skip_db_insert: If True, skip database insertion
            start_page: Optional first 1-indexed page to process
            end_page: Optional last 1-indexed page to process
            all_pages: If True, process cover/map/summary pages too
            
        Returns:
            Dictionary with processing results
        """
        pdf_path = Path(pdf_path)
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        
        pdf_name = pdf_path.stem
        start_time = time.time()
        results = {
            'pdf_name': pdf_name,
            'total_pages': 0,
            'pages_processed': 0,
            'total_records': 0,
            'records_inserted': 0,
            'records_duplicated': 0,
            'database_enabled': bool(self.db_manager and not skip_db_insert),
            'errors': [],
            'pages': []
        }
        
        json_logger.info(f"Starting PDF processing: {pdf_name}")
        
        try:
            total_pages = ImageProcessor.get_pdf_page_count(str(pdf_path))
            results['total_pages'] = total_pages

            effective_start_page = start_page
            effective_end_page = end_page
            if not all_pages:
                if effective_start_page is None and total_pages > 2:
                    effective_start_page = 3
                if effective_end_page is None and total_pages > 3:
                    effective_end_page = total_pages - 1

            results['start_page'] = effective_start_page or 1
            results['end_page'] = effective_end_page or total_pages

            # Step 1: Convert only the selected PDF pages to images. This is
            # critical on EC2 because 600 DPI full-PDF rasterization is slow.
            json_logger.info(
                "Converting PDF to images...",
                start_page=results['start_page'],
                end_page=results['end_page'],
            )
            pages_dir = work_dir / "pages"
            image_paths = ImageProcessor.convert_pdf_to_images(
                str(pdf_path),
                str(pages_dir),
                dpi=600,
                first_page=results['start_page'],
                last_page=results['end_page'],
            )
            
            # Step 2-5: Process voter-list pages by default. Electoral roll
            # PDFs usually have cover/map pages before records and a summary
            # page at the end; use all_pages or explicit ranges to override.
            for image_path in image_paths:
                page_number = int(Path(image_path).stem.rsplit("_", 1)[1])
                try:
                    page_result = self._process_page(
                        image_path,
                        page_number,
                        pdf_name,
                        work_dir
                    )
                    
                    results['pages_processed'] += 1
                    results['pages'].append(page_result)
                    results['total_records'] += page_result['record_count']
                    
                    json_logger.info(
                        f"Processed page {page_number}",
                        page_number=page_number,
                        records=page_result['record_count']
                    )
                    
                except Exception as e:
                    error_msg = f"Error processing page {page_number}: {str(e)}"
                    results['errors'].append(error_msg)
                    json_logger.error(error_msg, page_number=page_number, exception=e)
        
        except Exception as e:
            error_msg = f"PDF processing failed: {str(e)}"
            results['errors'].append(error_msg)
            json_logger.error(error_msg, exception=e)

        self._fill_missing_serial_numbers(results['pages'])

        if not skip_db_insert and self.db_manager:
            for page_result in results['pages']:
                inserted, skipped = self.db_manager.insert_batch(page_result['records'])
                results['records_inserted'] += inserted
                results['records_duplicated'] += skipped
        
        results['processing_time_seconds'] = time.time() - start_time
        json_logger.info(
            "PDF processing completed",
            pages_processed=results['pages_processed'],
            total_records=results['total_records'],
            processing_time=results['processing_time_seconds']
        )
        
        return results

    @staticmethod
    def _fill_missing_serial_numbers(pages: List[Dict]) -> None:
        """
        Fill serial numbers from the card sequence when enough anchors exist.

        Electoral roll cards are printed in strict reading order. OCR often
        misses the small serial in the card header, so we infer only when the
        observed records agree on the same sequence offset.
        """
        anchors = []
        for page in pages:
            for record in page.get('records', []):
                serial = record.get('serial_number')
                page_number = record.get('page_number')
                card_number = record.get('card_number')
                if not serial or not page_number or not card_number:
                    continue
                position = (page_number * 30) + card_number
                anchors.append(serial - position)

        if not anchors:
            return

        offsets = Counter(anchors)
        offset, count = offsets.most_common(1)[0]
        if count < 2 and len(offsets) > 1:
            return

        for page in pages:
            for record in page.get('records', []):
                if record.get('serial_number'):
                    continue
                page_number = record.get('page_number')
                card_number = record.get('card_number')
                if not page_number or not card_number:
                    continue
                inferred = (page_number * 30) + card_number + offset
                if 1 <= inferred <= 9999:
                    record['serial_number'] = inferred
    
    def _process_page(
        self,
        image_path: str,
        page_number: int,
        pdf_name: str,
        work_dir: Path
    ) -> Dict:
        """Process single page"""
        page_start = time.time()
        page_result = {
            'page_number': page_number,
            'record_count': 0,
            'records': [],
            'processing_time': 0
        }
        
        # Step 2: Enhance image
        enhanced_dir = work_dir / "enhanced"
        enhanced_dir.mkdir(parents=True, exist_ok=True)
        enhanced_path = enhanced_dir / f"page_{page_number:04d}.jpg"
        
        ImageProcessor.enhance_image(image_path, str(enhanced_path))
        
        # Step 3: Detect and crop cards
        cards_dir = work_dir / "cards"
        cards_dir.mkdir(parents=True, exist_ok=True)
        
        boxes = CardDetector.detect_cards(str(enhanced_path))
        card_paths = CardDetector.crop_cards(str(enhanced_path), boxes, str(cards_dir))
        json_logger.info(
            "Detected card crops",
            page_number=page_number,
            cards=len(card_paths),
        )
        
        # Step 4-5: OCR and parse cards (parallel)
        page_result['records'] = self._process_cards_parallel(
            card_paths,
            page_number,
            pdf_name
        )
        
        page_result['record_count'] = len(page_result['records'])
        page_result['processing_time'] = time.time() - page_start
        
        return page_result
    
    def _process_cards_parallel(
        self,
        card_paths: List[str],
        page_number: int,
        pdf_name: str
    ) -> List[Dict]:
        """Process cards in parallel"""
        records = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._ocr_and_parse_card, card_path, idx + 1, page_number, pdf_name): idx
                for idx, card_path in enumerate(card_paths)
            }
            
            for future in as_completed(futures):
                try:
                    record = future.result()
                    if record:
                        records.append(record)
                except Exception as e:
                    json_logger.error(f"Card processing error: {str(e)}", exception=e)
        
        return sorted(records, key=lambda record: record.get('card_number') or 0)
    
    def _ocr_and_parse_card(
        self,
        card_path: str,
        card_number: int,
        page_number: int,
        pdf_name: str
    ) -> Optional[Dict]:
        """OCR and parse single card"""
        try:
            # Step 4: OCR
            json_logger.info(
                "OCR card started",
                page_number=page_number,
                card_number=card_number,
                exhaustive_ocr=self.exhaustive_ocr,
            )
            text, confidence = OCREngine.extract_text(
                card_path,
                language='eng',
                exhaustive=self.exhaustive_ocr,
                timeout=self.ocr_timeout,
            )
            
            if not text or confidence < 20:
                json_logger.warning(
                    f"Low confidence OCR for card {card_number}",
                    page_number=page_number,
                    card_number=card_number,
                    confidence=confidence
                )
                return None
            
            # Step 5: Parse
            voter_record = NLPParser.parse_voter_card(text, confidence)
            
            # Step 6: Validate
            if not self._validate_record(voter_record):
                json_logger.warning(
                    f"Validation failed for card {card_number}",
                    page_number=page_number,
                    card_number=card_number
                )
                return None
            
            record_dict = voter_record.to_dict()
            record_dict['pdf_name'] = pdf_name
            record_dict['page_number'] = page_number
            record_dict['card_number'] = card_number
            
            return record_dict
            
        except Exception as e:
            json_logger.error(
                f"Card OCR failed: {str(e)}",
                page_number=page_number,
                card_number=card_number,
                exception=e
            )
            return None
    
    @staticmethod
    def _validate_record(record: VoterRecord) -> bool:
        """Validate voter record"""
        # Age validation
        if record.age is not None and not (18 <= record.age <= 120):
            return False
        
        # Gender validation
        if record.gender and record.gender not in ['Male', 'Female', 'Third Gender']:
            return False
        
        # EPIC validation
        if record.epic_number:
            import re
            if not re.match(r'^[A-Z]{3}\d{7}$', record.epic_number):
                return False
        
        # At least name or EPIC should be present
        if not record.name and not record.epic_number:
            return False
        
        return True
