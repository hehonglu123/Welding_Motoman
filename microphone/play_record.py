import wave, copy
import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
# TODO: Read metadata for samplerate and channels
samplerate = 44000
channels = 1


wavfile = wave.open('../../debug_data/weld_scan_job200_v52023_07_26_12_51_35/layer_60/mic_recording.wav', 'rb')

print ("nsamples:", wavfile.getnframes())
print ("sample size:", wavfile.getsampwidth())
print ("channels:", wavfile.getnchannels())

audio_data=np.frombuffer(wavfile.readframes(wavfile.getnframes()),dtype=np.int16)
plt.plot(audio_data)
plt.show()
audio_data_float=copy.deepcopy(audio_data).astype(np.float32).reshape((-1,576))
audio_data_float/=32767
###play back
with sd.OutputStream(channels=channels, samplerate=samplerate) as sd_stream:
    for i in range(len(audio_data_float)):
        sd_stream.write(audio_data_float[i])
