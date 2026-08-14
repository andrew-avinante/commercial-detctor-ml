import cv2
import librosa
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
# Video processing constants
VIDEO_PATH = '/mnt/a/MediaProcessing/Converted/The ScoobyDoo Show/The Tar Monster s03e05.mp4'
# VIDEO_PATH = '/mnt/g/Projects/python/commercial-detector-ml/pre_train_vids/train_1387.mp4'
BATCH_SIZE_SECONDS = 8
FRAME_RATE = 24
WINDOW_OVERLAP = 4
TOTAL_FRAMES = BATCH_SIZE_SECONDS * FRAME_RATE
WINDOW_FRAMES = WINDOW_OVERLAP * FRAME_RATE
VIDEO_OUTPUT_SIZE = (128, 128)

AUDIO_BATCH_SIZE = 2000
SAMPLING_RATE = 48000

def preprocess_video_batch(frames):
    processed_frames = []
    for frame in frames:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Resize the frame to 128x128
        gray_frame = cv2.resize(gray_frame, VIDEO_OUTPUT_SIZE)
        # Normalize the pixel values to be between 0 and 1
        gray_frame = gray_frame / 255.0
        processed_frames.append(gray_frame)
    # Pad the frames with white frames if the batch size is less than the specified batch size
    white_frames = TOTAL_FRAMES - len(processed_frames)
    if white_frames > 0:
        for i in range(white_frames):
            white_frame = np.ones(VIDEO_OUTPUT_SIZE)
            processed_frames.append(white_frame)
    processed_frames = np.array(processed_frames)
    
    return np.reshape(processed_frames, (TOTAL_FRAMES, -1))

def preprocess_audio_batch(audio):
    audio = librosa.util.normalize(audio)
    audio = librosa.util.fix_length(audio, size=TOTAL_FRAMES, axis=0)
    return audio

def predict_video(video, audio, model):
    audio = audio.reshape((1, BATCH_SIZE_SECONDS * FRAME_RATE, 1))
    predictions = model.predict([video, audio])
    
    return predictions


cap = cv2.VideoCapture(VIDEO_PATH)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
count = 1
final_result = []
model_file = 'trained_model.h5'
model = tf.keras.models.load_model(model_file, custom_objects={'KerasLayer':hub.KerasLayer})
print(frame_count, TOTAL_FRAMES)
for batch_start in range(0, frame_count, WINDOW_FRAMES):
    # Set the video file pointer to the start of the current batch
    cap.set(cv2.CAP_PROP_POS_FRAMES, batch_start)
    
    # Read the current batch of frames
    frames = []
    for i in range(TOTAL_FRAMES):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    # Preprocess the video batch
    processed_frames = preprocess_video_batch(frames)
    
    # Preprocess the audio batch
    audio, sr = librosa.load(VIDEO_PATH, sr=FRAME_RATE, offset=(count - 1) * WINDOW_OVERLAP, duration=BATCH_SIZE_SECONDS)
    processed_audio = preprocess_audio_batch(audio)

    if len(processed_frames) > len(processed_audio):
        processed_frames = processed_frames[:len(processed_audio)]
    elif len(processed_audio) > len(processed_frames):
        processed_audio = processed_audio[:len(processed_audio)]
    
    processed_audio = np.array([processed_audio])
    processed_frames = np.array([processed_frames])

    predictions = predict_video(processed_frames, processed_audio, model)

    local_result = list(list(np.argmax(predictions, axis=2))[0])

    n = len(local_result)
    overlap = n // 2

    if not final_result:
        final_result += local_result
    else:
        if 1 in local_result[:overlap]:
            final_result[-overlap:] = local_result[:overlap]  # Replace the overlap
        final_result += local_result[overlap:]  # Append the non-overlapping part

    print('Scanned: {:02}:{:02}'.format((count * WINDOW_OVERLAP) // 60, (count * WINDOW_OVERLAP) % 60))
    count += 1
    # Do whatever you want with the processed video and audio data
    # e.g., save them to disk, pass them to a neural network, etc.

isFade = False
count = 0

for i in final_result:
    count += 1
    if i == 1 and isFade == False:
        start = count / 24
        print(f"Start: {start}, Time: {count}")
        isFade = True
    elif i == 0 and isFade == True:
        print(f"End: {count / 24}, Time: {count}")
        isFade = False

np.save('frames.npy', np.array(final_result))
        