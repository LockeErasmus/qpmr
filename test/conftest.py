"""
Pytest configuration
"""
import pytest

import os
import sys

repository_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
source_path = os.path.join(repository_path, "src")
#sys.path.append(repository_path)
sys.path.append(source_path)