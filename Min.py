import numpy as np

def minimum(I):
    m = min(np.array(I).flatten())
    return m

# Example usage
image_data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
result = minimum(image_data)
print(result)

