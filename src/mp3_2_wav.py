from os import path
import os
from pydub import AudioSegment

target_dir='train/'
for file in os.listdir(target_dir):
    if file.endswith('.mp3'):
        f_name,_ = path.splitext(file)
        sound = AudioSegment.from_mp3(target_dir+file)
        sound.export(target_dir+ f_name + '.wav', format="wav")
