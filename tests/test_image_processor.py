"""Unit tests for image processing"""
import pytest
import cv2
import numpy as np
from pathlib import Path
import tempfile

from src.image_processor import ImageProcessor
from src.card_detector import CardDetector


class TestImageProcessor:
    """Test image processing pipeline"""
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample image"""
        img = np.ones((600, 800, 3), dtype=np.uint8) * 255
        return img
    
    def test_deskew(self, sample_image):
        """Test deskew"""
        result = ImageProcessor.deskew(sample_image)
        assert result.shape == sample_image.shape
    
    def test_clahe(self, sample_image):
        """Test CLAHE"""
        result = ImageProcessor.apply_clahe(sample_image)
        assert result.shape == sample_image.shape
    
    def test_denoise(self, sample_image):
        """Test denoising"""
        result = ImageProcessor.denoise(sample_image)
        assert result.shape == sample_image.shape
    
    def test_sharpen(self, sample_image):
        """Test sharpening"""
        result = ImageProcessor.sharpen(sample_image)
        assert result.shape == sample_image.shape


class TestCardDetector:
    """Test card detection and cropping"""
    
    def test_organize_grid(self):
        """Test grid organization"""
        boxes = [
            (10, 10, 100, 150),   # Top-left
            (120, 10, 100, 150),  # Top-center
            (230, 10, 100, 150),  # Top-right
        ]
        organized = CardDetector._organize_grid(boxes)
        assert len(organized) == 3
        
        # Should be ordered left to right
        assert organized[0][0] < organized[1][0] < organized[2][0]
