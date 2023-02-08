import ffmpeg
import re
from subprocess import Popen
import random
import subprocess

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

            padding = self.fps * 2
            extra = random.randint(0, 1)

            if len(matches):
                if mode == SCREEN:
                    result.append(
                        {
                            'start': (round(float(matches[0]) * self.fps)) - padding - (1 if extra else 0),
                            'end': (round(float(re.findall(REGEX[mode]['end'], decoded_line)[0]) * self.fps)) + padding + (1 if not extra else 0)
                        })
                else:
                    stack.append(float(matches[0]))
            elif len(matches := re.findall(REGEX[mode]['end'], decoded_line)):
                result.append({
                    'start': (round(stack.pop()) * self.fps) - padding - (1 if extra else 0),
                    'end': (round(float(matches[0])) * self.fps) + padding + (1 if not extra else 0)
                })

        return result

    def detect_null_av(self, file: str, mode: str, result: list = [], **kwargs: dict):
        stream = ffmpeg.input(file)
        stream = ffmpeg.filter(stream, filter_name='blackdetect' if mode == SCREEN else 'silencedetect', **kwargs)
        stream = ffmpeg.output(stream, "/dev/null", format="rawvideo" if mode == SCREEN else 'null')
        return self._extract_times(ffmpeg.run_async(stream, overwrite_output=True, pipe_stderr=True), mode, result)

    def get_black_spots(self, file: str, result: list):
        return self.detect_null_av(file, SCREEN, result, d=0.05, pix_th=0.1)

parser = ChapterParser(24)
results = []
input_file = '/mnt/a/MediaProcessing/Converted/The ScoobyDoo Show/A Bum Steer for Scooby s01e11.mp4'
parser.get_black_spots(input_file, results)

start = 0
count = start

for i in results:
    output_video = f'training_vids/train_{count}.mp4'
    if i['start'] < 0:
        continue
    frames = [i['end'] - i['start']]
    start = i['start'] / 24
    end = i['end'] / 24

    command = f"ffmpeg -ss {start} -to {end} -i '{input_file}' -vf scale=128:128 -codec:a copy -crf 23 -r 24 '{output_video}'"

    subprocess.run(command, shell=True)

    count += 1
