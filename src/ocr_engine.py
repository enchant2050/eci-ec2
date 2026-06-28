"""OCR Engine using Tesseract 5 LSTM"""
import pytesseract
from pytesseract import Output
import logging
from typing import Dict, List, Tuple, Optional
import re

logger = logging.getLogger(__name__)


class OCREngine:
    """Tesseract 5 LSTM-based OCR with confidence scoring"""
    
    # Tesseract configuration
    CUSTOM_CONFIG = r'--oem 3 --psm 6'
    LANGUAGES = ['eng']  # Add support for Indian languages later
    
    @staticmethod
    def extract_text(image_path: str, language: str = 'eng') -> Tuple[str, float]:
        """
        Extract text from image using Tesseract 5
        
        Args:
            image_path: Path to voter card image
            language: Language code (eng, hin, mar, etc.)
            
        Returns:
            Tuple of (extracted_text, average_confidence)
        """
        try:
            # Configure language support
            lang_config = language
            if language != 'eng':
                lang_config = f'eng+{language}'
            
            # Extract text with confidence scores
            data = pytesseract.image_to_data(
                image_path,
                lang=lang_config,
                config=OCREngine.CUSTOM_CONFIG,
                output_type=Output.DICT
            )
            
            # Get overall text
            text = pytesseract.image_to_string(
                image_path,
                lang=lang_config,
                config=OCREngine.CUSTOM_CONFIG
            )
            
            # Tesseract returns confidence values under the "conf" key. Values
            # can be "-1" or decimal strings depending on the installed build.
            confidences = []
            for conf in data.get('conf', []):
                try:
                    value = float(conf)
                except (TypeError, ValueError):
                    continue
                if value > 0:
                    confidences.append(value)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            return text.strip(), avg_confidence
            
        except Exception as e:
            logger.error(f"OCR failed for {image_path}: {str(e)}")
            return "", 0.0
    
    @staticmethod
    def extract_text_regions(
        image_path: str,
        language: str = 'eng'
    ) -> List[Dict]:
        """
        Extract text with bounding boxes and confidence scores
        
        Returns:
            List of dicts with text, bbox, and confidence
        """
        try:
            lang_config = language
            if language != 'eng':
                lang_config = f'eng+{language}'
            
            data = pytesseract.image_to_data(
                image_path,
                lang=lang_config,
                config=OCREngine.CUSTOM_CONFIG,
                output_type=Output.DICT
            )
            
            results = []
            for i in range(len(data['text'])):
                text = data['text'][i].strip()
                if text:
                    try:
                        confidence = float(data['conf'][i])
                    except (TypeError, ValueError):
                        confidence = 0.0
                    results.append({
                        'text': text,
                        'confidence': confidence,
                        'bbox': {
                            'x': int(data['left'][i]),
                            'y': int(data['top'][i]),
                            'width': int(data['width'][i]),
                            'height': int(data['height'][i])
                        }
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"OCR region extraction failed for {image_path}: {str(e)}")
            return []
