"""Voter Card Detection and Cropping"""
import cv2
import numpy as np
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)


class CardDetector:
    """Detect and crop individual voter cards from pages"""
    
    EXPECTED_ROWS = 10
    EXPECTED_COLS = 3
    
    @staticmethod
    def detect_cards(image_path: str) -> List[Tuple[int, int, int, int]]:
        """
        Detect voter card regions (bounding boxes) in image
        
        Returns:
            List of (x, y, w, h) tuples for each detected card
        """
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Dilate to connect nearby edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated = cv2.dilate(edges, kernel, iterations=3)
        
        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area and aspect ratio
        h, w = image.shape[:2]
        min_area = (w * h) / (30 * 2)  # Approximate card area
        max_area = (w * h) / 20
        
        boxes = []
        for contour in contours:
            x, y, cw, ch = cv2.boundingRect(contour)
            area = cw * ch
            aspect_ratio = float(cw) / ch if ch > 0 else 0
            
            # Card aspect ratio approximately 2:3
            if min_area < area < max_area and 0.5 < aspect_ratio < 0.8:
                boxes.append((x, y, cw, ch))
        
        # Sort boxes: top-to-bottom, then left-to-right
        boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
        
        # Organize into grid (10 rows, 3 cols)
        organized_boxes = CardDetector._organize_grid(boxes)
        
        return organized_boxes
    
    @staticmethod
    def _organize_grid(boxes: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
        """Organize detected boxes into 10x3 grid"""
        if not boxes:
            return []
        
        # Group by approximate y-coordinate (row)
        rows = {}
        row_threshold = 50  # pixels
        
        for box in boxes:
            x, y, w, h = box
            row_key = None
            
            for existing_y in rows.keys():
                if abs(y - existing_y) < row_threshold:
                    row_key = existing_y
                    break
            
            if row_key is None:
                row_key = y
            
            if row_key not in rows:
                rows[row_key] = []
            rows[row_key].append(box)
        
        # Sort rows by y-coordinate
        sorted_rows = sorted(rows.items())
        
        # Sort boxes within each row by x-coordinate
        result = []
        for _, row_boxes in sorted_rows:
            row_boxes = sorted(row_boxes, key=lambda b: b[0])
            result.extend(row_boxes)
        
        return result
    
    @staticmethod
    def crop_cards(
        image_path: str,
        boxes: List[Tuple[int, int, int, int]],
        output_dir: str
    ) -> List[str]:
        """
        Crop individual voter cards from image
        
        Args:
            image_path: Input image path
            boxes: List of bounding boxes
            output_dir: Directory to save cropped cards
            
        Returns:
            List of output card image paths
        """
        from pathlib import Path
        
        image = cv2.imread(image_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract page number from filename
        page_name = Path(image_path).stem  # e.g., "page_0003"
        
        output_paths = []
        for i, (x, y, w, h) in enumerate(boxes, 1):
            # Add padding
            padding = 10
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(image.shape[1] - x, w + 2 * padding)
            h = min(image.shape[0] - y, h + 2 * padding)
            
            card = image[y:y+h, x:x+w]
            
            output_path = output_dir / f"{page_name}_card_{i:03d}.jpg"
            cv2.imwrite(str(output_path), card)
            output_paths.append(str(output_path))
        
        logger.info(f"Cropped {len(output_paths)} cards from {image_path}")
        return output_paths
