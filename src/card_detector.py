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
    MIN_EXPECTED_CARDS = 20
    
    @staticmethod
    def detect_cards(image_path: str) -> List[Tuple[int, int, int, int]]:
        """
        Detect voter card regions (bounding boxes) in image
        
        Returns:
            List of (x, y, w, h) tuples for each detected card
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Dilate to connect nearby edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated = cv2.dilate(edges, kernel, iterations=3)
        
        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area and aspect ratio. ECI voter cards are laid
        # out as landscape cells in a 3x10 grid on the page.
        h, w = image.shape[:2]
        min_area = (w * h) / 120
        max_area = (w * h) / 15
        
        boxes = []
        for contour in contours:
            x, y, cw, ch = cv2.boundingRect(contour)
            area = cw * ch
            aspect_ratio = float(cw) / ch if ch > 0 else 0
            
            if min_area < area < max_area and 1.1 < aspect_ratio < 3.2:
                boxes.append((x, y, cw, ch))
        
        # Sort boxes: top-to-bottom, then left-to-right
        boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
        
        organized_boxes = CardDetector._organize_grid(boxes)

        if len(organized_boxes) >= CardDetector.MIN_EXPECTED_CARDS:
            return organized_boxes[: CardDetector.EXPECTED_ROWS * CardDetector.EXPECTED_COLS]

        logger.warning(
            "Contour card detection found %s cards; falling back to page grid",
            len(organized_boxes),
        )
        return CardDetector._fallback_grid(gray, allow_content_fallback=len(organized_boxes) >= 6)

    @staticmethod
    def _fallback_grid(
        gray: np.ndarray,
        allow_content_fallback: bool = True,
    ) -> List[Tuple[int, int, int, int]]:
        """Infer the standard 3x10 card grid when contour detection is weak."""
        h, w = gray.shape[:2]

        # Detect long ruling lines first; those give the most reliable grid
        # bounds on scanned electoral roll pages.
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(30, w // 18), 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(30, h // 45)))
        horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
        vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)

        y_positions = CardDetector._line_positions(horizontal.sum(axis=1), threshold=255 * max(20, w // 8))
        x_positions = CardDetector._line_positions(vertical.sum(axis=0), threshold=255 * max(20, h // 12))

        if len(y_positions) >= CardDetector.EXPECTED_ROWS + 1 and len(x_positions) >= CardDetector.EXPECTED_COLS + 1:
            x_lines = CardDetector._select_grid_lines(x_positions, CardDetector.EXPECTED_COLS + 1)
            y_lines = CardDetector._select_grid_lines(y_positions, CardDetector.EXPECTED_ROWS + 1)
            if x_lines and y_lines:
                return CardDetector._boxes_from_lines(x_lines, y_lines, w, h)

        if not allow_content_fallback:
            return []

        # Last resort for damaged scans: use the dark-pixel content bounds and
        # split into the expected ECI page grid.
        content = binary > 0
        ys, xs = np.where(content)
        if len(xs) == 0 or len(ys) == 0:
            return []

        x1 = max(0, int(np.percentile(xs, 1)))
        x2 = min(w - 1, int(np.percentile(xs, 99)))
        y1 = max(0, int(np.percentile(ys, 8)))
        y2 = min(h - 1, int(np.percentile(ys, 99)))

        # Header text often sits above the cards. Start the grid after the
        # first substantial horizontal rule below the top eighth of the page.
        lower_lines = [y for y in y_positions if y > h * 0.12]
        if lower_lines:
            y1 = max(y1, lower_lines[0])

        cell_w = (x2 - x1) / CardDetector.EXPECTED_COLS
        cell_h = (y2 - y1) / CardDetector.EXPECTED_ROWS
        boxes = []
        for row in range(CardDetector.EXPECTED_ROWS):
            for col in range(CardDetector.EXPECTED_COLS):
                x = int(round(x1 + col * cell_w))
                y = int(round(y1 + row * cell_h))
                next_x = int(round(x1 + (col + 1) * cell_w))
                next_y = int(round(y1 + (row + 1) * cell_h))
                boxes.append((x, y, max(1, next_x - x), max(1, next_y - y)))
        return boxes

    @staticmethod
    def _line_positions(projection: np.ndarray, threshold: int) -> List[int]:
        """Return center positions for contiguous high-projection line runs."""
        positions = np.where(projection > threshold)[0]
        if len(positions) == 0:
            return []

        groups = []
        start = int(positions[0])
        previous = int(positions[0])
        for pos in positions[1:]:
            pos = int(pos)
            if pos - previous > 3:
                groups.append((start, previous))
                start = pos
            previous = pos
        groups.append((start, previous))
        return [(start + end) // 2 for start, end in groups]

    @staticmethod
    def _select_grid_lines(positions: List[int], count: int) -> List[int]:
        """Choose a consecutive set of grid lines with consistent spacing."""
        if len(positions) < count:
            return []

        best = []
        best_score = float("inf")
        for idx in range(0, len(positions) - count + 1):
            candidate = positions[idx: idx + count]
            gaps = np.diff(candidate)
            if np.any(gaps <= 0):
                continue
            score = float(np.std(gaps) / max(np.mean(gaps), 1))
            if score < best_score:
                best_score = score
                best = candidate
        return best

    @staticmethod
    def _boxes_from_lines(
        x_lines: List[int],
        y_lines: List[int],
        image_width: int,
        image_height: int,
    ) -> List[Tuple[int, int, int, int]]:
        boxes = []
        for row in range(CardDetector.EXPECTED_ROWS):
            for col in range(CardDetector.EXPECTED_COLS):
                x1 = max(0, x_lines[col])
                y1 = max(0, y_lines[row])
                x2 = min(image_width - 1, x_lines[col + 1])
                y2 = min(image_height - 1, y_lines[row + 1])
                boxes.append((x1, y1, max(1, x2 - x1), max(1, y2 - y1)))
        return boxes
    
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
