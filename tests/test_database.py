"""Unit tests for database"""
import pytest
import tempfile
from pathlib import Path

from src.database import DatabaseManager, ElectoralRollRecord


class TestDatabaseManager:
    """Test database operations"""
    
    @pytest.fixture
    def db_manager(self):
        """Create in-memory SQLite database for testing"""
        # Using SQLite for testing instead of PostgreSQL
        db_url = "sqlite:///:memory:"
        manager = DatabaseManager(db_url)
        manager.create_tables()
        return manager
    
    def test_insert_record(self, db_manager):
        """Test record insertion"""
        session = db_manager.SessionLocal()
        record = {
            'pdf_name': 'test.pdf',
            'page_number': 3,
            'card_number': 1,
            'serial_number': 1,
            'epic_number': 'PRI3186269',
            'name': 'TEST NAME',
            'age': 45,
            'gender': 'Male',
        }
        
        result = db_manager.insert_record(session, record)
        assert result is True
        session.close()
    
    def test_duplicate_epic(self, db_manager):
        """Test duplicate EPIC handling"""
        session = db_manager.SessionLocal()
        record = {
            'pdf_name': 'test.pdf',
            'page_number': 3,
            'card_number': 1,
            'epic_number': 'PRI3186269',
            'name': 'TEST NAME',
        }
        
        # First insert should succeed
        assert db_manager.insert_record(session, record) is True
        
        # Second insert should fail due to unique constraint
        assert db_manager.insert_record(session, record) is False
        session.close()
