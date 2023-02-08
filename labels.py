import cv2
import librosa
import numpy as np

# Convert the video and audio features to numpy arrays
TOTAL_FRAMES = 192

data = [
    {
        "start": "00:00:01,369",
        "end": "00:00:03,012"
    },
    {
        "start": "00:00:01,315",
        "end": "00:00:03,026"
    },
    {
        "start": "00:00:01,513",
        "end": "00:00:04,861"
    },
    {
        "start": "00:00:01,323",
        "end": "00:00:03,572"
    },
    {
        "start": "00:00:01,474",
        "end": "00:00:05,192"
    },
    {
        "start": "00:00:01,126",
        "end": "00:00:03,318"
    },
    {
        "start": "00:00:01,302",
        "end": "00:00:03,401"
    },
    {
        "start": "00:00:01,248",
        "end": "00:00:04,246"
    }
]

def convert_timestamp(timestamp):
    timestamp = timestamp.replace(',', '.')
    hours, minutes, seconds = map(float, timestamp.split(':'))
    return round((hours * 3600 + minutes * 60 + seconds) * 24)

offset = 8

for i, j in enumerate(data):
    start = convert_timestamp(j['start'])
    end = convert_timestamp(j['end'].replace(',', '.'))

    video_features = np.array([0] * start + [1] * (end - start) + [0] * (TOTAL_FRAMES - end))

    # Save the features as numpy arrays
    np.save(f'training_dir/labels_{i + offset}.npy', video_features)
    print(len(video_features))
