# File: ./test/conftest.py

import sys
import os

# 将项目根目录添加到 sys.path
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.insert(0, root_path)
