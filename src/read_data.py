import os
import librosa
import librosa.display
import matplotlib.pyplot as plt

train_dir = '../res/train/'
train_clips = os.listdir(train_dir)
print("Number of train .wav files in audio folder:", len(train_clips))

x, sr = librosa.load(train_dir + train_clips[0], sr=None, mono=True)
print(type(x), type(sr))
print(x.shape, sr)

plt.figure(figsize=(14, 5))
librosa.display.waveplot(x, sr=sr)

# ======================================================================================================================
# ======================================================================================================================

test_dir = '../res/test/'
test_clips = os.listdir(train_dir)
print("Number of test .wav files in res/test folder:", len(test_clips))
