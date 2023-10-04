import wave, time, traceback, os
from RobotRaconteur.Client import *
import numpy as np
import matplotlib.pyplot as plt
# TODO: Read metadata for samplerate and channels
samplerate = 44100
channels = 1

audio_recording=[]
def new_frame(pipe_ep):
    global audio_recording
    #Loop to get the newest frame
    while (pipe_ep.Available > 0):
        #Receive the packet
        audio_recording.extend(pipe_ep.ReceivePacket().audio_data)

def save_frame(audio_recording,name):
    first_channel = np.concatenate(audio_recording)
    first_channel_int16=(first_channel*32767).astype(np.int16)
    with wave.open(name+'.wav', 'wb') as wav_file:
        # Set the WAV file parameters
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)  # 2 bytes per sample (16-bit)
        wav_file.setframerate(samplerate)
        # Write the audio data to the WAV file
        wav_file.writeframes(first_channel_int16.tobytes())

c = RRN.ConnectService('rr+tcp://127.0.0.1:60828?service=microphone')

p = c.microphone_stream.Connect(-1)
p.PacketReceivedEvent+=new_frame
os.makedirs('recoreded_data/',exist_ok=True)
counts=0
while True:
    audio_recording=[]
    time.sleep(20)
    try:
        save_frame(audio_recording,'recoreded_data/test'+str(counts))
        print('audio length: ',len(audio_recording)) 
        counts+=1
    except:
        traceback.print_exc()
        break
