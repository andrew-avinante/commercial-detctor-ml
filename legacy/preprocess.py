import cv2
import numpy as np
import glob
import librosa

TOTAL_FRAMES = 192
SAMPLING_RATE = 48000

def preprocess_video(video):
    # Load the video using OpenCV
    cap = cv2.VideoCapture(video)
    
    # Get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(total_frames)
    # Initialize a list to store the grayscale arrays of pixel values
    processed_frames = []
    white_frames = (TOTAL_FRAMES - total_frames)
    
    for i in range(total_frames):
        # Read a frame from the video
        ret, frame = cap.read()
        
        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Resize the frame to 128x128
        gray_frame = cv2.resize(gray_frame, (128, 128))
        
        # Normalize the pixel values to be between 0 and 1
        gray_frame = gray_frame / 255.0

        # Add the grayscale frame to the list of processed frames
        processed_frames.append(gray_frame)

    if white_frames > 0:

        for i in range(white_frames):
            white_frame = np.ones((128, 128))

            processed_frames.append(white_frame)

    else:
        print(f"Video: {video} contains {total_frames} frames which is over {TOTAL_FRAMES} frames")
        return None
    
    # Convert the list of processed frames into a numpy array
    processed_frames = np.array(processed_frames)
    
    return processed_frames, white_frames

def average_every_n(list, n):
    result = []
    for i in range(0, len(list), n):
        avg = sum(list[i:i+n]) / n
        result.append(avg)
    return result

def preprocess_audio(audio_path):
    # Load audio
    audio, sr = librosa.load(audio_path, sr=None)
    
    audio = average_every_n(audio, 2000)
    
    # Normalize the audio
    audio = (audio - np.mean(audio, axis=0, keepdims=True)) / np.std(audio, axis=0, keepdims=True)

    silence = TOTAL_FRAMES - len(audio)

    if silence > 0:
        audio = audio + [0] * silence

    return audio

def save_as_npy(video_frames, filename):
    # Save the preprocessed data as a .npy file
    np.save(filename, video_frames)

# Example usage
video_features_files = glob.glob('training_vids/*.mp4')

offset = 8

for i, video in enumerate(video_features_files):
    output_name = f'training_dir/processed_video_{i + offset}.npy'
    video_frames, white_frames = preprocess_video(video)
    save_as_npy(video_frames, output_name)
    print(f"Padded {video}:{output_name} with {white_frames} frames")
    print(video_frames.shape)
    output_name = f'training_dir/processed_audio_{i + offset}.npy'
    audio = preprocess_audio(video)
    save_as_npy(audio, output_name)

    print(f"Processed {video}:{output_name} with {len(audio)} samples")