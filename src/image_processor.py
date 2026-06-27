"""Image Processing Pipeline for Electoral Roll PDFs"""
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)


class ImageProcessor:
    """Handles PDF to image conversion and enhancement"""
    
    @staticmethod
    def convert_pdf_to_images(
        pdf_path: str,
        output_dir: str,
        dpi: int = 600
    ) -> List[str]:
        """
        Convert PDF to high-resolution JPEG images using pdf2image
        
        Args:
            pdf_path: Path to input PDF
            output_dir: Directory to save images
            dpi: Resolution in DPI
            
        Returns:
            List of output image paths
        """
        from pdf2image import convert_from_path
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        images = convert_from_path(pdf_path, dpi=dpi)
        output_paths = []
        
        for i, image in enumerate(images, 1):
            output_path = output_dir / f"page_{i:04d}.jpg"
            image.save(str(output_path), "JPEG", quality=95)
            output_paths.append(str(output_path))
            
        logger.info(f"Converted PDF to {len(images)} images at {dpi} DPI")
        return output_paths
    
    @staticmethod
    def deskew(image: np.ndarray) -> np.ndarray:
        """Correct image rotation/skew"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect edges
        edges = cv2.Canny(gray, 100, 200)
        
        # Hough line detection
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
        
        if lines is None or len(lines) == 0:
            return image
        
        # Calculate angle
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            angles.append(angle)
        
        median_angle = np.median(angles)
        
        # Rotate image
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        rotated = cv2.warpAffine(image, rotation_matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)
        
        return rotated
    
    @staticmethod
    def apply_clahe(image: np.ndarray) -> np.ndarray:
        """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    
    @staticmethod
    def adaptive_threshold(image: np.ndarray) -> np.ndarray:
        """Apply adaptive thresholding"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        threshold = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=11,
            C=2
        )
        return cv2.cvtColor(threshold, cv2.COLOR_GRAY2BGR)
    
    @staticmethod
    def denoise(image: np.ndarray) -> np.ndarray:
        """Remove noise using bilateral filtering and morphological operations"""
        # Bilateral filter to preserve edges
        denoised = cv2.bilateralFilter(image, 9, 75, 75)
        
        # Morphological opening (erosion followed by dilation)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        opened = cv2.morphologyEx(denoised, cv2.MORPH_OPEN, kernel)
        
        # Median blur
        denoised = cv2.medianBlur(opened, 5)
        
        return denoised
    
    @staticmethod
    def sharpen(image: np.ndarray) -> np.ndarray:
        """Sharpen image"""
        kernel = np.array([
            [-1, -1, -1],
            [-1,  9, -1],
            [-1, -1, -1]
        ]) / 1.0
        sharpened = cv2.filter2D(image, -1, kernel)
        return sharpened
    
    @classmethod
    def enhance_image(cls, image_path: str, output_path: str) -> None:
        """
        Complete image enhancement pipeline
        
        Args:
            image_path: Input image path
            output_path: Output image path
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")
        
        # Pipeline: Deskew -> CLAHE -> Denoise -> Adaptive Threshold -> Sharpen
        image = cls.deskew(image)
        image = cls.apply_clahe(image)
        image = cls.denoise(image)
        image = cls.adaptive_threshold(image)
        image = cls.sharpen(image)
        
        cv2.imwrite(output_path, image)
        logger.info(f"Enhanced image saved: {output_path}")
