import numpy as np
from scipy.ndimage import minimum_filter

def airlight(haze_img, wsz):
    A = np.zeros(3)

    for k in range(3):
        min_img = minimum_filter(haze_img[:, :, k], size=wsz, mode='reflect')
        A[k] = np.max(min_img)

    return A


