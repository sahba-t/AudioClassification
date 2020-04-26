from os import path
import os
from pydub import AudioSegment

def mp3_to_wav(dir_name):
    if not dir_name.endswith("/"):
        dir_name = dir_name + "/"
    target_dir = dir_name
    for file in os.listdir(target_dir):
        if file.endswith('.mp3'):
            f_name, _ = path.splitext(file)
            sound = AudioSegment.from_mp3(target_dir+file)
            sound.export(target_dir+ f_name + '.wav', format="wav")

def print_files(dir_name= '../res/train/'):
    for i, file in enumerate(os.listdir(dir_name)):
        print(file)
        if i == 10:
            break