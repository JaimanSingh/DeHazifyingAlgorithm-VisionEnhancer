import numpy as np
from PIL import Image

def boxfilter(imSrc, r):
    hei, wid = imSrc.shape
    imDst = np.zeros_like(imSrc, dtype=np.float64)

    imCum = np.cumsum(imSrc, axis=0)
    imDst[0:r+1, :] = imCum[1:r+1, :]
    imDst[r+1:hei-r, :] = imCum[2*r+1:hei, :] - imCum[0:hei-2*r-1, :]
    imDst[hei-r:hei, :] = np.tile(imCum[hei-1, :], (r, 1)) - imCum[hei-2*r-1:hei-r-1, :]

    imCum = np.cumsum(imDst, axis=1)
    imDst[:, 0:r+1] = imCum[:, 1:r+1]
    imDst[:, r+1:wid-r] = imCum[:, 2*r+1:wid] - imCum[:, 0:wid-2*r-1]
    imDst[:, wid-r:wid] = np.tile(imCum[:, wid-1], (1, r)) - imCum[:, wid-2*r-1:wid-r-1]

    return imDst

# Example usage
image_path = 'input_image.jpg'
radius = 3  # Example radius value, replace with your actual value

try:
    image = np.array(Image.open(image_path).convert('L'))  # Read the image and convert to grayscale
    result = boxfilter(image, radius)
except Exception as e:
    print(f"Error: {e}")

