import glob
import random
import json
import os

prefix = 'training'

# Example usage
video_features_files = glob.glob(f'{prefix}_vids/*.mp4')

labels = glob.glob('label_json/*.json')

data = []

for l in labels:
    with open(l, 'r') as f:
        data += json.load(f)

    random.shuffle(data)

for i, d in enumerate(data):
    num = d['num']
    d['num'] = i
    old_filename = f'training_vids/train_{num:05}.mp4'
    new_filename = f'shuffled_dir/train_{i:05}.mp4'
    os.rename(old_filename, new_filename)

with open('label_json/tmp.json', 'w') as f:
        json.dump(data, f)
