# -*- coding: utf-8 -*-
"""pytest 根配置：把项目根加入 sys.path，使 `import config` / `import src.*` 生效。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
