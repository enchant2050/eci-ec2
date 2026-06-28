"""NLP Parser for Electoral Roll Data Extraction"""
import re
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from rapidfuzz import fuzz
import logging

logger = logging.getLogger(__name__)


@dataclass
class VoterRecord:
    """Structured voter record"""
    serial_number: Optional[int] = None
    epic_number: Optional[str] = None
    name: Optional[str] = None
    relation_type: Optional[str] = None  # Father, Husband, Mother
    relation_name: Optional[str] = None
    house_number: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    confidence: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'serial_number': self.serial_number,
            'epic_number': self.epic_number,
            'name': self.name,
            'relation_type': self.relation_type,
            'relation_name': self.relation_name,
            'house_number': self.house_number,
            'age': self.age,
            'gender': self.gender,
            'ocr_confidence': round(self.confidence, 2)
        }


class NLPParser:
    """Extract structured data from OCR text"""
    
    # Dictionary for OCR error correction
    OCR_CORRECTIONS = {
        'houae': 'house',
        'houso': 'house',
        'gendar': 'gender',
        'neme': 'name',
        'relision': 'religion',
        'husbamd': 'husband',
        'fathor': 'father',
        'mothor': 'mother',
        'age': 'age',
        'aje': 'age',
        'aga': 'age',
        'fathers': 'father',
        'husbands': 'husband',
        'mothers': 'mother',
    }
    
    RELATION_TYPES = {
        'F': 'Father',
        'H': 'Husband',
        'M': 'Mother',
        'FATHER': 'Father',
        'HUSBAND': 'Husband',
        'MOTHER': 'Mother',
    }
    
    GENDERS = {
        'M': 'Male',
        'F': 'Female',
        'MALE': 'Male',
        'FEMALE': 'Female',
        'TG': 'Third Gender',
        'THIRD GENDER': 'Third Gender',
    }
    
    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalize and clean text"""
        # Remove extra whitespace
        text = ' '.join(text.split())
        # Remove special characters except common ones
        text = re.sub(r'[^a-zA-Z0-9\s\-/.:\']', '', text)
        return text.strip()
    
    @staticmethod
    def correct_ocr_errors(text: str) -> str:
        """Correct common OCR errors"""
        text = text.lower()
        for wrong, correct in NLPParser.OCR_CORRECTIONS.items():
            text = re.sub(r'\b' + wrong + r'\b', correct, text, flags=re.IGNORECASE)
        return text
    
    @staticmethod
    def extract_serial_number(text: str) -> Optional[int]:
        """Extract serial number (typically 1-189)"""
        # Look for the first small number at the start of a card/line.
        matches = re.findall(r'^\s*(\d{1,4})\b', text, re.MULTILINE)
        if matches:
            num = int(matches[0])
            if 1 <= num <= 9999:
                return num
        return None
    
    @staticmethod
    def extract_epic_number(text: str) -> Optional[str]:
        """Extract EPIC number (format: XXX0000000)"""
        compact = re.sub(r'[^A-Z0-9]', '', text.upper())
        matches = re.findall(r'([A-Z]{3}\d{7})', compact)
        if matches:
            return matches[0]
        return None
    
    @staticmethod
    def extract_gender(text: str) -> Optional[str]:
        """Extract gender"""
        text = text.upper().strip()

        label_match = re.search(r'GENDER\s*[:=\-]?\s*(THIRD\s+GENDER|MALE|FEMALE|TG|M|F)\b', text)
        if label_match:
            text = label_match.group(1)

        for key in sorted(NLPParser.GENDERS, key=len, reverse=True):
            value = NLPParser.GENDERS[key]
            if re.search(r'\b' + re.escape(key) + r'\b', text):
                return value
        
        # Fuzzy matching
        for key, value in NLPParser.GENDERS.items():
            if fuzz.ratio(text, key) > 85:
                return value
        
        return None
    
    @staticmethod
    def extract_relation_type(text: str) -> Optional[str]:
        """Extract relation type (Father, Husband, Mother)"""
        text = text.upper().replace("'", "").strip()
        
        for key, value in NLPParser.RELATION_TYPES.items():
            if key in text:
                return value
        
        # Fuzzy matching
        for key, value in NLPParser.RELATION_TYPES.items():
            if fuzz.ratio(text, key) > 80:
                return value
        
        return None
    
    @staticmethod
    def extract_age(text: str) -> Optional[int]:
        """Extract age (18-120)"""
        matches = re.findall(r'\b(\d{1,3})\b', text)
        for match in matches:
            age = int(match)
            if 18 <= age <= 120:
                return age
        return None
    
    @staticmethod
    def extract_house_number(text: str) -> Optional[str]:
        """Extract house number (various formats: 1, 1A, 12/3)"""
        text = text.strip()

        label_match = re.search(r'house\s*(?:number|no\.?)?\s*[:=\-]?\s*([A-Za-z0-9/\-.]+)', text, re.IGNORECASE)
        if label_match:
            return label_match.group(1).strip(".-")
        
        # Match patterns like: 1, 1A, 12/3, 1-A
        matches = re.findall(r'\b(\d+[A-Z]?(?:/\d+)?)\b', text)
        if matches:
            return matches[0]
        
        return None
    
    @staticmethod
    def extract_name(text: str) -> Optional[str]:
        """Extract name (remove prefixes/suffixes)"""
        text = NLPParser.normalize_text(text)
        text = NLPParser.correct_ocr_errors(text)
        
        # Remove common prefixes
        text = re.sub(r'^(Name|EPIC|Age|Gender|House|House Number|Relation|Serial)\s*[:=\-]?\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'^(Father|Mother|Husband)(?:s)?(?: Name)?\s*[:=\-]?\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\b(?:Age|Gender|House(?: Number)?)\b.*$', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\b[A-Z]{3}\s*\d{7}\b', '', text, flags=re.IGNORECASE)
        text = ' '.join(text.split()).strip(" .:-")
        
        # Should be all uppercase or title case
        if len(text) > 3 and len(text) < 100:
            return text.upper()
        
        return None
    
    @staticmethod
    def parse_voter_card(ocr_text: str, confidence: float = 0.0) -> VoterRecord:
        """
        Parse OCR text into structured VoterRecord
        
        Args:
            ocr_text: Raw OCR output from Tesseract
            confidence: OCR confidence score
            
        Returns:
            VoterRecord object
        """
        record = VoterRecord(confidence=confidence)
        
        lines = [line.strip() for line in ocr_text.strip().split('\n') if line.strip()]
        full_text = "\n".join(lines)

        record.serial_number = NLPParser.extract_serial_number(full_text)
        record.epic_number = NLPParser.extract_epic_number(full_text)
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Extract components
            if not record.serial_number:
                sn = NLPParser.extract_serial_number(line)
                if sn:
                    record.serial_number = sn
                    continue
            
            if not record.epic_number:
                record.epic_number = NLPParser.extract_epic_number(line)
            
            if re.search(r'\bname\b', line, re.IGNORECASE) and not record.name:
                parts = re.split(r'[:=\-]', line, 1)
                if len(parts) == 2:
                    name = NLPParser.extract_name(parts[1])
                    if name:
                        record.name = name
                        continue
            
            if re.search(r'\b(father|husband|mother)', line, re.IGNORECASE):
                # Relation type
                rt = NLPParser.extract_relation_type(line)
                if rt and not record.relation_type:
                    record.relation_type = rt
                
                # Relation name
                parts = re.split(r'[:=\-]', line, 1)
                if len(parts) == 2:
                    rn = NLPParser.extract_name(parts[1])
                    if rn and not record.relation_name:
                        record.relation_name = rn
            
            if 'house' in line.lower() and not record.house_number:
                parts = re.split(r'[:=\-]', line, 1)
                if len(parts) == 2:
                    hn = NLPParser.extract_house_number(parts[1])
                    if hn:
                        record.house_number = hn
                        continue
            
            if 'age' in line.lower() and not record.age:
                age = NLPParser.extract_age(line)
                if age:
                    record.age = age
            
            if 'gender' in line.lower() and not record.gender:
                gender = NLPParser.extract_gender(line)
                if gender:
                    record.gender = gender
                    continue

        if not record.name:
            for line in lines:
                if re.search(r'\b(father|husband|mother|house|age|gender|epic)\b', line, re.IGNORECASE):
                    continue
                candidate = NLPParser.extract_name(line)
                if candidate and not re.fullmatch(r'\d+', candidate):
                    record.name = candidate
                    break
        
        return record
