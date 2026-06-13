"""
pytest conftest: adds the project root to sys.path so all
`from app.xxx import yyy` imports work when running pytest from
anywhere inside the workmate/ directory.
"""
import sys
import os

# Insert the project root (workmate/) at the front of sys.path
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
