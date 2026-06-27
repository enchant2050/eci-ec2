"""Database Models and Operations"""
from typing import Tuple
from sqlalchemy import create_engine, Column, String, Integer, Float, DateTime, UniqueConstraint
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from datetime import datetime
import uuid
import logging

logger = logging.getLogger(__name__)

Base = declarative_base()


class ElectoralRollRecord(Base):
    """Electoral roll voter record"""
    __tablename__ = 'electoral_roll_records'
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    pdf_name = Column(String(255), nullable=False)
    page_number = Column(Integer, nullable=False)
    card_number = Column(Integer, nullable=False)
    serial_number = Column(Integer, nullable=True)
    epic_number = Column(String(20), nullable=True)
    name = Column(String(255), nullable=True)
    relation_type = Column(String(50), nullable=True)  # Father, Husband, Mother
    relation_name = Column(String(255), nullable=True)
    house_number = Column(String(50), nullable=True)
    age = Column(Integer, nullable=True)
    gender = Column(String(20), nullable=True)  # Male, Female, Third Gender
    ocr_confidence = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Ensure unique EPIC numbers
    __table_args__ = (
        UniqueConstraint('epic_number', name='uq_epic_number'),
    )
    
    def to_dict(self):
        return {
            'id': self.id,
            'pdf_name': self.pdf_name,
            'page_number': self.page_number,
            'card_number': self.card_number,
            'serial_number': self.serial_number,
            'epic_number': self.epic_number,
            'name': self.name,
            'relation_type': self.relation_type,
            'relation_name': self.relation_name,
            'house_number': self.house_number,
            'age': self.age,
            'gender': self.gender,
            'ocr_confidence': round(self.ocr_confidence, 2),
            'created_at': self.created_at.isoformat() if self.created_at else None,
        }


class DatabaseManager:
    """Database operations"""
    
    def __init__(self, connection_string: str):
        """
        Initialize database connection
        
        Args:
            connection_string: PostgreSQL connection string
                              e.g., postgresql://user:pass@host:5432/dbname
        """
        self.engine = create_engine(connection_string, pool_pre_ping=True)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
    
    def create_tables(self) -> None:
        """Create all tables"""
        Base.metadata.create_all(bind=self.engine)
        logger.info("Database tables created")
    
    def insert_record(self, session: Session, record: dict) -> bool:
        """
        Insert voter record
        
        Args:
            session: SQLAlchemy session
            record: Dictionary with voter data
            
        Returns:
            True if successful, False on duplicate EPIC
        """
        try:
            db_record = ElectoralRollRecord(
                pdf_name=record.get('pdf_name'),
                page_number=record.get('page_number'),
                card_number=record.get('card_number'),
                serial_number=record.get('serial_number'),
                epic_number=record.get('epic_number'),
                name=record.get('name'),
                relation_type=record.get('relation_type'),
                relation_name=record.get('relation_name'),
                house_number=record.get('house_number'),
                age=record.get('age'),
                gender=record.get('gender'),
                ocr_confidence=record.get('ocr_confidence', 0.0),
            )
            session.add(db_record)
            session.commit()
            return True
        except Exception as e:
            session.rollback()
            if "unique constraint" in str(e).lower():
                logger.warning(f"Duplicate EPIC number: {record.get('epic_number')}")
                return False
            logger.error(f"Database insert error: {str(e)}")
            raise
    
    def insert_batch(self, records: list) -> Tuple[int, int]:
        """
        Insert batch of records
        
        Args:
            records: List of record dictionaries
            
        Returns:
            Tuple of (inserted_count, skipped_count)
        """
        session = self.SessionLocal()
        inserted = 0
        skipped = 0
        
        try:
            for record in records:
                if self.insert_record(session, record):
                    inserted += 1
                else:
                    skipped += 1
        finally:
            session.close()
        
        return inserted, skipped
    
    def get_record_by_epic(self, epic_number: str) -> ElectoralRollRecord:
        """Get record by EPIC number"""
        session = self.SessionLocal()
        try:
            return session.query(ElectoralRollRecord).filter_by(epic_number=epic_number).first()
        finally:
            session.close()
    
    def get_records_by_page(self, pdf_name: str, page_number: int) -> list:
        """Get all records for a page"""
        session = self.SessionLocal()
        try:
            return session.query(ElectoralRollRecord).filter_by(
                pdf_name=pdf_name,
                page_number=page_number
            ).all()
        finally:
            session.close()


from typing import Tuple
