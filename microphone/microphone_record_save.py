import wave
from RobotRaconteur.Client import *
import numpy as np
import matplotlib.pyplot as plt
# TODO: Read metadata for samplerate and channels
samplerate = 44000
channels = 1


c = RRN.ConnectService('rr+tcp://localhost:60828?service=microphone')

print("Begin recording")
recording = c.capture_microphone(2)
print("End recording")

audio=[]
for a in recording:
    for audio_data in a.audio_data:
        audio.append(a.audio_data)


first_channel = np.concatenate([a1[0] for a1 in audio])

first_channel_int16=(first_channel*32767).astype(np.int16)
plt.plot(first_channel_int16)
plt.show()
print(first_channel.shape)

with wave.open('output.wav', 'wb') as wav_file:
    # Set the WAV file parameters
    wav_file.setnchannels(channels)
    wav_file.setsampwidth(2)  # 2 bytes per sample (16-bit)
    wav_file.setframerate(samplerate)

    # Write the audio data to the WAV file
    wav_file.writeframes(first_channel_int16.tobytes())

print('saving finished')
