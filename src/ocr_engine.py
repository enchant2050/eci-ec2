"""OCR Engine using Tesseract 5 LSTM"""
import cv2
import pytesseract
from pytesseract import Output
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


class OCREngine:
    """Tesseract 5 LSTM-based OCR with confidence scoring"""
    
    # Tesseract configuration
    CUSTOM_CONFIG = r'--oem 3 --psm 6'
    OCR_CONFIGS = (
        r'--oem 3 --psm 6',
        r'--oem 3 --psm 4',
        r'--oem 3 --psm 11',
    )
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
            
            attempts = []
            for image in OCREngine._image_variants(image_path):
                for config in OCREngine.OCR_CONFIGS:
                    text, confidence = OCREngine._run_tesseract(image, lang_config, config)
                    if text:
                        attempts.append((text, confidence))

            if not attempts:
                return "", 0.0

            merged_text = OCREngine._merge_text_attempts([text for text, _ in attempts])
            best_confidence = max(confidence for _, confidence in attempts)
            return merged_text, best_confidence
            
        except Exception as e:
            logger.error(f"OCR failed for {image_path}: {str(e)}")
            return "", 0.0

    @staticmethod
    def _image_variants(image_path: str) -> List:
        """Build OCR-friendly card image variants without writing temp files."""
        image = cv2.imread(image_path)
        if image is None:
            return [image_path]

        variants = [image]
        scale = 2
        enlarged = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        variants.append(enlarged)

        gray = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)
        gray = cv2.fastNlMeansDenoising(gray, h=10)
        threshold = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=31,
            C=10,
        )
        variants.append(threshold)
        return variants

    @staticmethod
    def _run_tesseract(image, lang_config: str, config: str) -> Tuple[str, float]:
        data = pytesseract.image_to_data(
            image,
            lang=lang_config,
            config=config,
            output_type=Output.DICT
        )
        text = pytesseract.image_to_string(
            image,
            lang=lang_config,
            config=config
        )
        return text.strip(), OCREngine._average_confidence(data)

    @staticmethod
    def _average_confidence(data: Dict) -> float:
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
        return sum(confidences) / len(confidences) if confidences else 0.0

    @staticmethod
    def _merge_text_attempts(texts: List[str]) -> str:
        """Keep unique OCR lines from multiple passes, preserving order."""
        lines = []
        seen = set()
        for text in texts:
            for line in text.splitlines():
                normalized = ' '.join(line.split())
                if not normalized:
                    continue
                key = normalized.lower()
                if key in seen:
                    continue
                seen.add(key)
                lines.append(normalized)
        return "\n".join(lines)
    
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
