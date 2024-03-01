import os
import cv2

path = 'input_frames/'
output_path = 'output_frames/'

image_files_in = [f for f in os.listdir(path) if f.endswith('.png')]

for l, current_in_filename in enumerate(image_files_in, start=1):
    current_in_path = os.path.join(path, current_in_filename)
    
    I = cv2.imread(current_in_path)
    r, t = bounding_function(I, 4)
    
    print(l)
    
    output_filename = os.path.join(output_path, current_in_filename)
    cv2.imwrite(output_filename, r)
