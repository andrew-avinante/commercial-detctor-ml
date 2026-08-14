import glob
import os
import json

pre_train_vids = glob.glob('training_vids/*.mp4')
start = len(glob.glob('training_vids/*.mp4'))
offset = start
labels = []


for f in pre_train_vids:
    file_name = 'training_vids/train_{:05}.mp4'.format(offset)
    os.rename(f, file_name)
    labels.append( {
        "num": offset,
        "start": "00:00:00",
        "end": "00:00:00"
    })
    offset += 1

with open('label_json/{:05}-{:05}.json'.format(start, offset - 1), 'w') as f:
    json.dump(labels, f)
