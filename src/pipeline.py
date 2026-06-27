"""Main OCR Pipeline Orchestration"""
import os
import json
from pathlib import Path
from typing import List, Dict, Optional
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    
    def __init__(self, db_connection_string: Optional[str] = None, max_workers: int = 4):
        """
        Initialize pipeline
        
        Args:
            db_connection_string: PostgreSQL connection string
            max_workers: Number of parallel workers for card OCR
        """
        self.db_manager = None
        if db_connection_string:
            self.db_manager = DatabaseManager(db_connection_string)
            self.db_manager.create_tables()
        
        self.max_workers = max_workers
    
    def process_pdf(
        self,
        pdf_path: str,
        work_dir: str = "/tmp/electoral_roll_ocr",
        skip_db_insert: bool = False
    ) -> Dict:
        """
        Process entire PDF document
        
        Args:
            pdf_path: Path to input PDF
            work_dir: Working directory for temporary files
            skip_db_insert: If True, skip database insertion
            
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
            'errors': [],
            'pages': []
        }
        
        json_logger.info(f"Starting PDF processing: {pdf_name}")
        
        try:
            # Step 1: Convert PDF to images
            json_logger.info("Converting PDF to images...")
            pages_dir = work_dir / "pages"
            image_paths = ImageProcessor.convert_pdf_to_images(
                str(pdf_path),
                str(pages_dir),
                dpi=600
            )
            results['total_pages'] = len(image_paths)
            
            # Step 2-5: Process each page (skip first 2 pages and last page)
            for page_num, image_path in enumerate(image_paths):
                # Skip metadata (page 1), maps (page 2), and summary (last page)
                if page_num < 2 or page_num == len(image_paths) - 1:
                    continue
                
                try:
                    page_result = self._process_page(
                        image_path,
                        page_num + 1,  # 1-indexed
                        pdf_name,
                        work_dir
                    )
                    
                    results['pages_processed'] += 1
                    results['pages'].append(page_result)
                    results['total_records'] += page_result['record_count']
                    
                    # Insert into database
                    if not skip_db_insert and self.db_manager:
                        inserted, skipped = self.db_manager.insert_batch(page_result['records'])
                        results['records_inserted'] += inserted
                        results['records_duplicated'] += skipped
                    
                    json_logger.info(
                        f"Processed page {page_num + 1}",
                        page_number=page_num + 1,
                        records=page_result['record_count']
                    )
                    
                except Exception as e:
                    error_msg = f"Error processing page {page_num + 1}: {str(e)}"
                    results['errors'].append(error_msg)
                    json_logger.error(error_msg, page_number=page_num + 1, exception=e)
        
        except Exception as e:
            error_msg = f"PDF processing failed: {str(e)}"
            results['errors'].append(error_msg)
            json_logger.error(error_msg, exception=e)
        
        results['processing_time_seconds'] = time.time() - start_time
        json_logger.info(
            "PDF processing completed",
            pages_processed=results['pages_processed'],
            total_records=results['total_records'],
            processing_time=results['processing_time_seconds']
        )
        
        return results
    
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
        
        return records
    
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
            text, confidence = OCREngine.extract_text(card_path, language='eng')
            
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
