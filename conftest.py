import sys
import os

# Add repo root to sys.path so tests can import src.*
root_dir = os.path.dirname(os.path.abspath(__file__))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
