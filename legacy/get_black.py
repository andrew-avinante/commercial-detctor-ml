import ffmpeg
import re
from subprocess import Popen
import random
import subprocess
import glob

TOTAL_FRAMES = 192
SCREEN = "SCREEN"
AUDIO = "AUDIO"

REGEX = {
    SCREEN: {
        "start": r"(?<=black_start:)[0-9]*\.?[0-9]*",
        "end": r"(?<=black_end:)[0-9]*\.?[0-9]*"
    },
    AUDIO: {
        "start": r"(?<=silence_start: )[0-9]*\.?[0-9]*",
        "end": r"(?<=silence_end: )[0-9]*\.[0-9]*"
    }
}

import random

def generate_random_clip(start_time, end_time, total_duration):
    # Calculate the duration of the black detect segment
    black_duration = end_time - start_time
    
    # Calculate the maximum amount of padding that can be added to the beginning and end of the black detect segment
    max_padding_duration = total_duration - black_duration

    clip_start = start_time - max_padding_duration / 2
    clip_end = end_time + max_padding_duration / 2

    if clip_start < 0:
        clip_start = 0
        clip_end = total_duration
    
    return clip_start, clip_end



class ChapterParser:
    def __init__(self, fps: int, start_threshold: int = 60, end_threshold: int = 60):
        self.fps = fps
        self.start_threshold = start_threshold
        self.end_threshold = end_threshold


    def _extract_times(self, process: Popen, mode: str, result: list = []):
        stack = []

        while True:
            line = process.stderr.readline()
            if not line:
                break
            decoded_line = line.decode('utf-8')
            matches = re.findall(REGEX[mode]['start'], decoded_line)

            if len(matches):
                if mode == SCREEN:
                    result.append(
                        {
                            'start': (round(float(matches[0]) * self.fps)),
                            'end': (round(float(re.findall(REGEX[mode]['end'], decoded_line)[0]) * self.fps))
                        })
                else:
                    stack.append(float(matches[0]))
            elif len(matches := re.findall(REGEX[mode]['end'], decoded_line)):
                result.append({
                    'start': (round(stack.pop()) * self.fps),
                    'end': (round(float(matches[0])) * self.fps)
                })

        return result

    def detect_null_av(self, file: str, mode: str, result: list = [], **kwargs: dict):
        stream = ffmpeg.input(file)
        stream = ffmpeg.filter(stream, filter_name='blackdetect' if mode == SCREEN else 'silencedetect', **kwargs)
        stream = ffmpeg.output(stream, "/dev/null", format="rawvideo" if mode == SCREEN else 'null')
        return self._extract_times(ffmpeg.run_async(stream, overwrite_output=True, pipe_stderr=True), mode, result)

    def get_black_spots(self, file: str, result: list):
        return self.detect_null_av(file, SCREEN, result, d=1, pix_th=0.25)

parser = ChapterParser(24)

# input_file = '/mnt/a/MediaProcessing/Converted/The ScoobyDoo Show/To Switch a Witch s03e04.mp4'

files = glob.glob('/mnt/a/MediaProcessing/Converted/*/*.mp4')

start = 0
count = start

for f in files:
    results = []
    episode_name = f.rsplit('/', 1)[1].split('.')[0]
    print(episode_name)
    parser.get_black_spots(f, results)

# results = [{'start': i - 4 * 24, 'end': i + 3 * 24} for i in [random.uniform(24 * 8, 20 * 60 * 24) for _ in range(10)]]

    for i in results:
        output_video = f'pre_train_vids/train_{count:05}_{episode_name}.mp4'
        if i['start'] < 0:
            continue
        frames = [i['end'] - i['start']]
        start = i['start'] / 24
        end = i['end'] / 24

        start, end = generate_random_clip(start, end, 8)
        print(start, end)
        command = f"ffmpeg -ss {start} -to {end} -i '{f}' -vf scale=128:128 -c:v libx264 -crf 23 -r 24 '{output_video}'"

        subprocess.run(command, shell=True)

        count += 1

print(f'Created {count} videos')