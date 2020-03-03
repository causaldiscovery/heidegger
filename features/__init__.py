
import os
from os.path import basename
currentPath = os.path.dirname(__file__)
__all__ = [os.path.splitext(basename(f))[0] for f in os.listdir(currentPath)]