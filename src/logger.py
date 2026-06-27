"""Structured JSON logger for OCR processing."""
import json
import logging
import sys
from datetime import datetime
from typing import Any, Dict, Optional
import traceback


class JSONLogger:
    """Structured JSON logging for the OCR application."""
    
    def __init__(self, service_name: str = "electoral-roll-ocr"):
        self.service_name = service_name
        self.logger = logging.getLogger(service_name)
        self.logger.setLevel(logging.DEBUG)
        
    def _log(
        self,
        level: str,
        message: str,
        page_number: Optional[int] = None,
        card_number: Optional[int] = None,
        processing_time: Optional[float] = None,
        confidence: Optional[float] = None,
        error: Optional[str] = None,
        retries: int = 0,
        **extra: Any
    ) -> None:
        """Internal logging method"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": level,
            "service": self.service_name,
            "message": message,
        }
        
        if page_number is not None:
            log_entry["page_number"] = page_number
        if card_number is not None:
            log_entry["card_number"] = card_number
        if processing_time is not None:
            log_entry["processing_time_seconds"] = round(processing_time, 2)
        if confidence is not None:
            log_entry["confidence"] = round(confidence, 2)
        if error is not None:
            log_entry["error"] = error
        if retries > 0:
            log_entry["retries"] = retries
            
        log_entry.update(extra)
        print(json.dumps(log_entry, default=str))
        
    def info(self, message: str, **kwargs) -> None:
        self._log("INFO", message, **kwargs)
        
    def debug(self, message: str, **kwargs) -> None:
        self._log("DEBUG", message, **kwargs)
        
    def warning(self, message: str, **kwargs) -> None:
        self._log("WARNING", message, **kwargs)
        
    def error(self, message: str, exception: Optional[Exception] = None, **kwargs) -> None:
        error_msg = str(exception) if exception else kwargs.get("error", "Unknown error")
        if exception:
            error_msg += f"\n{traceback.format_exc()}"
        self._log("ERROR", message, error=error_msg, **kwargs)


# Global logger instance
logger = JSONLogger("electoral-roll-ocr")
