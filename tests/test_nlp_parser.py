"""Unit tests for NLP parser"""
import pytest

from src.nlp_parser import NLPParser, VoterRecord


class TestNLPParser:
    """Test NLP parsing"""
    
    def test_normalize_text(self):
        """Test text normalization"""
        text = "Hello   World  !!!"
        result = NLPParser.normalize_text(text)
        assert result == "Hello World"
    
    def test_extract_serial_number(self):
        """Test serial number extraction"""
        text = "1"
        result = NLPParser.extract_serial_number(text)
        assert result == 1
        
        text = "189"
        result = NLPParser.extract_serial_number(text)
        assert result == 189
        
        text = "500"
        result = NLPParser.extract_serial_number(text)
        assert result == 500
    
    def test_extract_epic_number(self):
        """Test EPIC number extraction"""
        text = "Name : PULIKAR KUJUR\nEPIC: PRI3186269"
        result = NLPParser.extract_epic_number(text)
        assert result == "PRI3186269"

        text = "123 PRI 3186269"
        result = NLPParser.extract_epic_number(text)
        assert result == "PRI3186269"

        text = "EPIC: PR13719739"
        result = NLPParser.extract_epic_number(text)
        assert result == "PRI3719739"
    
    def test_extract_gender(self):
        """Test gender extraction"""
        text = "Gender: Male"
        result = NLPParser.extract_gender(text)
        assert result == "Male"

        text = "Gender: Female"
        result = NLPParser.extract_gender(text)
        assert result == "Female"
        
        text = "F"
        result = NLPParser.extract_gender(text)
        assert result == "Female"
    
    def test_extract_age(self):
        """Test age extraction"""
        text = "Age: 45"
        result = NLPParser.extract_age(text)
        assert result == 45
        
        text = "Age: 200"
        result = NLPParser.extract_age(text)
        assert result is None
    
    def test_extract_house_number(self):
        """Test house number extraction"""
        text = "House Number: 12"
        result = NLPParser.extract_house_number(text)
        assert result == "12"
        
        text = "House: 12/3"
        result = NLPParser.extract_house_number(text)
        assert result == "12/3"
    
    def test_validate_voter_record(self):
        """Test record validation"""
        from src.pipeline import OCRPipeline
        
        record = VoterRecord(
            name="Test Name",
            age=45,
            gender="Male",
            epic_number="PRI3186269"
        )
        assert OCRPipeline._validate_record(record)
        
        # Invalid age
        record.age = 150
        assert not OCRPipeline._validate_record(record)
        
        # Invalid gender
        record.age = 45
        record.gender = "Unknown"
        assert not OCRPipeline._validate_record(record)

    def test_parse_noisy_eci_card(self):
        """Test common OCR layout from an ECI voter card."""
        text = """
        123 PRI 3186269
        Name : PULIKAR KUJUR
        Father's Name : MANGRA KUJUR
        House Number : 12/3
        Age : 45 Gender : Female
        """
        record = NLPParser.parse_voter_card(text, confidence=87.55)

        assert record.serial_number == 123
        assert record.epic_number == "PRI3186269"
        assert record.name == "PULIKAR KUJUR"
        assert record.relation_type == "Father"
        assert record.relation_name == "MANGRA KUJUR"
        assert record.house_number == "12/3"
        assert record.age == 45
        assert record.gender == "Female"
        assert record.to_dict()["ocr_confidence"] == 87.55
