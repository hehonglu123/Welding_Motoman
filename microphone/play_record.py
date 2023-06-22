import wave, copy
import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
# TODO: Read metadata for samplerate and channels
samplerate = 44000
channels = 1


wavfile = wave.open('output.wav', 'rb')

print ("nsamples:", wavfile.getnframes())
print ("sample size:", wavfile.getsampwidth())
print ("channels:", wavfile.getnchannels())

audio_data=np.frombuffer(wavfile.readframes(wavfile.getnframes()),dtype=np.int16)
audio_data_float=copy.deepcopy(audio_data).astype(np.float32).reshape((-1,576))
audio_data_float/=32767
###play back
with sd.OutputStream(channels=channels, samplerate=samplerate) as sd_stream:
    for i in range(len(audio_data_float)):
        sd_stream.write(audio_data_float[i])
