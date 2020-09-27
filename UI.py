import numpy as np
import cv2
from matplotlib import pyplot as plt
# Load an color image in grayscale
img = cv2.imread('HSBC.JPG')
img = cv2.resize(img, (100, 100))
mp = np.array(img)
print(mp)
cv2.imshow('output',img)

cv2.waitKey(0)
cv2.destroyAllWindows()