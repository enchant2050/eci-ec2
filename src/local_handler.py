"""Local OCR Handler for Testing"""
import json
from pathlib import Path
from typing import Dict, Any

from src.pipeline import OCRPipeline
from src.logger import logger as json_logger


def process_pdf_locally(
    pdf_path: str,
    work_dir: str = "/tmp/electoral_roll_ocr",
    db_connection: str = None,
    output_json_path: str = None
) -> Dict[str, Any]:
    """
    Process PDF locally (for development/testing)
    
    Args:
        pdf_path: Path to input PDF
        work_dir: Working directory for temporary files
        db_connection: Optional database connection string
        output_json_path: Optional path to save results as JSON
        
    Returns:
        Processing results
    """
    json_logger.info(f"Starting local PDF processing: {pdf_path}")
    
    pipeline = OCRPipeline(db_connection_string=db_connection)
    result = pipeline.process_pdf(pdf_path, work_dir=work_dir)
    
    # Save results
    if output_json_path:
        output_json_path = Path(output_json_path)
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_json_path, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        
        json_logger.info(f"Results saved to {output_json_path}")
    
    return result
