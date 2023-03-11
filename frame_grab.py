import cv2
import random
import glob

files = glob.glob('/mnt/a/MediaProcessing/Converted/The ScoobyDoo Show/*.mp4')

total = 0

for f in files:
    # Load video
    cap = cv2.VideoCapture(f)

    # Get total number of frames
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Generate random frame numbers
    frame_nums = random.sample(range(num_frames), 50)

    # Loop through selected frames and save as JPEGs
    for frame_num in frame_nums:
        # Set the frame number to read
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        
        # Read the frame
        ret, frame = cap.read()
        
        # Save as JPEG
        cv2.imwrite(f'train_frames/frame_{total}.jpg', frame)
        total += 1

    # Release video capture object
    cap.release()