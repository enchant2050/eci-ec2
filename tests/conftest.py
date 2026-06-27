"""Pytest configuration to ensure src module is discoverable."""
import sys
from pathlib import Path

# Add parent directory to path so 'src' can be imported
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
