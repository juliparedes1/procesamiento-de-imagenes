from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).parent

img = cv2.imread(str(BASE_DIR / "grade_sheet_1.png"), cv2.IMREAD_GRAYSCALE)
