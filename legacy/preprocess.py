import cv2
import numpy as np
import glob
import librosa
import warnings
import json
import re
warnings.filterwarnings("ignore")


TOTAL_FRAMES = 192
SAMPLING_RATE = 48000
buckets = {}

def add_gausian_noise(frame, mean = 0, stddev = 10):
    noise = np.random.normal(mean, stddev, frame.shape).astype('uint8')
    noisy_frame = cv2.add(frame, noise)
    return noisy_frame

def preprocess_video(video, flip = False, noisy = False, orientation = 0):
    # Load the video using OpenCV
    cap = cv2.VideoCapture(video)
    
    # Get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames in buckets:
        buckets[total_frames] += 1
    else:
        buckets[total_frames] = 0

    # Initialize a list to store the grayscale arrays of pixel values
    processed_frames = []
    white_frames = (TOTAL_FRAMES - total_frames)
    
    for i in range(total_frames):
        # Read a frame from the video
        ret, frame = cap.read()

        if flip:
            frame = cv2.flip(frame, orientation)

        if noisy:
            frame = add_gausian_noise(frame)

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
        processed_frames = processed_frames[:192]
    # Convert the list of processed frames into a numpy array
    processed_frames = np.array(processed_frames)
    
    return np.reshape(processed_frames, (TOTAL_FRAMES, -1))

def preprocess_audio(audio_path, pitch_shift = False, steps = 0):
    # Load audio
    audio, sr = librosa.load(audio_path, sr=24)
    
    if pitch_shift:
        audio = librosa.effects.pitch_shift(audio, sr, n_steps=steps)

    # Normalize the audio
    audio = librosa.util.fix_length(audio, size=TOTAL_FRAMES, axis=0)
    audio = librosa.util.normalize(audio)
    return audio
    
def save_as_npy(video_frames, filename):
    # Save the preprocessed data as a .npy file
    np.save(filename, video_frames)

def convert_timestamp(timestamp):
    timestamp = timestamp.replace(',', '.')
    hours, minutes, seconds = map(float, timestamp.split(':'))
    return round((hours * 3600 + minutes * 60 + seconds) * 24)

def find_dict_by_key_value(my_list, key, value):
    for item in my_list:
        if item[key] == value:
            return item
    return None

def extract_number_from_filename(filename):
    pattern = r"(pruned_vids/train|training_vids/train|test_vids/test)_([0-9]+)\.mp4"
    match = re.search(pattern, filename)
    if match:
        return int(match.group(2))
    else:
        return None

prefix = 'training'

# Example usage
video_features_files = glob.glob(f'{prefix}_vids/*.mp4')

offset = 0

labels = glob.glob('label_json/*.json')
data = []

for label in labels:
    with open(label, 'r') as f:
        content = json.load(f)
        data.append(content)

data = [item for sublist in data for item in sublist]

augment = False
do_all = False
augmented = 0
commercial_block = 0
other = 0

for i, video in enumerate(video_features_files):
    num = extract_number_from_filename(video)
    label_data = find_dict_by_key_value(data, 'num', num)

    if label_data == None:
        print(f"{video} does not have a corresponding label")
        continue

    if 'prune' in label_data and label_data['prune']:
        print(f"This clip was pruned: {video}")
        continue
    
    video_output_name_body = '{}_dir/processed_video_{:03d}'.format(prefix, num)
    video_output_name = f'{video_output_name_body}.npy'
    video_frames = preprocess_video(video)

    if video_frames is None:
        print(f'--- {video} equated to None ---')
        continue

    audio_output_name_body = '{}_dir/processed_audio_{:03d}'.format(prefix, num)
    audio_output_name = f'{audio_output_name_body}.npy'
    audio = preprocess_audio(video)
    if len(video_frames) > len(audio):
        video_frames = video_frames[:len(audio)]
    elif len(audio) > len(video_frames):
        audio = audio[:len(video_frames)]
    save_as_npy(video_frames, video_output_name)
    save_as_npy(audio, audio_output_name)

    print(f"Processed {video} with {len(audio)} samples and {len(video_frames)} frames")

    if prefix == 'test':
        continue

    start = convert_timestamp(label_data['start'])
    end = convert_timestamp(label_data['end'])
    total = len(video_frames)

    labels = np.array([0] * start + [1] * (end - start) + [0] * (total - end))
    if total != len(list(labels)):
        print(num, total, len(list(labels)), len(video_frames))

    label_output_name_body = f'{prefix}_dir/labels_{num:03}'
    label_output_name = f'{label_output_name_body}.npy'

    save_as_npy(labels, label_output_name)

    if len(labels) != len(video_frames) != len(audio):
        print(f'{video} labels:{len(labels)} != video_frames:{len(video_frames)} != audio:{len(audio)}')

    if start != end or do_all:
        if augment:
            for i in range(0, 4):
                suffix = f"_augmented_{i}.npy"
                augmented_video_output_name = f"{video_output_name_body}{suffix}"
                augmented_audio_output_name = f"{audio_output_name_body}{suffix}"
                augmented_label_output_name = f"{label_output_name_body}{suffix}"
                video_frames = preprocess_video(video, True, i > 1,  i % 2)
                audio = preprocess_audio(video, True, 2 + i * 2)

                if len(video_frames) > len(audio):
                    video_frames = video_frames[:len(audio)]
                elif len(audio) > len(video_frames):
                    audio = audio[:len(video_frames)]

                save_as_npy(video_frames, augmented_video_output_name)
                save_as_npy(audio, augmented_audio_output_name)
                save_as_npy(labels, augmented_label_output_name)

                if len(labels) != len(video_frames) != len(audio):
                    print(f'Augmented: {video} labels:{len(labels)} != video_frames:{len(video_frames)} != audio:{len(audio)}')

        augmented += 1
    else:
        other += 1

print(f'Processed {len(data)} labels')
print(f'Augmented {augmented} files')
print(f'Other {other}')
print(buckets)