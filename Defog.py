import numpy as np
import matplotlib.pyplot as plt  # Import the required module

def defog(haze_img, t, A, delta):
    t = np.maximum(np.abs(t), 0.00001) ** delta
    
    if len(A) == 1:
        A = np.full((3, 1), A)
    
    R = (haze_img[:, :, 0] - A[0]) / t + A[0]
    R = np.maximum(R, 0)
    R = np.minimum(R, 1)
    
    G = (haze_img[:, :, 1] - A[1]) / t + A[1]
    G = np.maximum(G, 0)
    G = np.minimum(G, 1)
    
    B = (haze_img[:, :, 2] - A[2]) / t + A[2]
    B = np.maximum(B, 0)
    B = np.minimum(B, 1)
    
    r_img = np.stack([R, G, B], axis=2)
    return r_img

# Example usage
haze_image_path = 'haze_image.jpg'
transmission_map = np.random.rand(480, 640)  # Example transmission map, replace with your actual map
airlight = np.array([0.7, 0.5, 0.4])  # Example airlight, replace with your actual value
delta_value = 0.8  # Example delta value, replace with your actual value

haze_image = plt.imread(haze_image_path)
result_image = defog(haze_image, transmission_map, airlight, delta_value)
