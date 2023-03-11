import os
import json

# load the JSON data
with open('label_json/00000-00359.json') as f:
    data = json.load(f)

# filter out files with "prune" set to true
data = [item for item in data if not item.get('prune')]

# sort the remaining items by "num"
data.sort(key=lambda x: x['num'])

# rename the files
for i, item in enumerate(data):
    old_filename = f'training_vids/train_{item["num"]:05}.mp4'
    new_filename = f'pruned_dir/train_{i:05}.mp4'
    os.rename(old_filename, new_filename)
    item["num"] = i

# update the JSON data
data = [item for item in data if not item.get('prune')]
zero = 0
with open(f'{zero:05}-{len(data):05}.json', 'w') as f:
    json.dump(data, f, indent=4)
