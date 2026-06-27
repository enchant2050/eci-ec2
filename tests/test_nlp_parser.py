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
        assert result is None
    
    def test_extract_epic_number(self):
        """Test EPIC number extraction"""
        text = "Name : PULIKAR KUJUR\nEPIC: PRI3186269"
        result = NLPParser.extract_epic_number(text)
        assert result == "PRI3186269"
    
    def test_extract_gender(self):
        """Test gender extraction"""
        text = "Gender: Male"
        result = NLPParser.extract_gender(text)
        assert result == "Male"
        
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
