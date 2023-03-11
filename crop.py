import cv2
import random
import glob
import os

files = glob.glob('/mnt/g/Projects/Dreambooth-Stable-Diffusion-main/trainingImages/hannahbarberra/aesthetic/*.jpg')

for i, f in enumerate(files):
    # Read the image
    img = cv2.imread(f)
    
    # Crop the image to 512x512
    cropped = img[0:512, 0:512]
    
    # Save the cropped image as a new JPEG file
    cv2.imwrite(os.path.join('train_frames', f'frame_{i}.jpg'), cropped)
