

import pyaudio
import wave
import time
import os
import pandas as pd
import numpy as np
from scipy.io import wavfile
from datetime import date

SAMPLE_RATE = 16000
CHANNELS = 4
RECORD_SECONDS = 3
BUFFER_SIZE = 1024
SAMPLE_WIDTH = 2

p = pyaudio.PyAudio()

try:
    df = pd.read_csv('voice_data.csv')
except:
    df = pd.DataFrame(columns=["name","label"])
    df.to_csv('voice_data.csv',index=False)
    df = pd.read_csv('voice_data.csv')
    
stream = p.open(format= p.get_format_from_width(SAMPLE_WIDTH),
                channels=CHANNELS,
                rate= SAMPLE_RATE,
                input=True,
                output=True,
                input_device_index = 0,
                frames_per_buffer=BUFFER_SIZE)
                
label = []
name = []
try:
    i = df.index.stop+1
except:
    i=1
print(i)

today = date.today()

try:
    os.system(f'mkdir recordings/{today}')
except:
    pass
counter = 0
while counter<=55:
    try:
        
        stream.start_stream()
        frames0 = []
        frames1 = []
        frames2 = []
        frames3 = []
        print('SPEAK')
        
        for j in range(0, int(SAMPLE_RATE/BUFFER_SIZE*RECORD_SECONDS)):
            data = stream.read(BUFFER_SIZE)
            a = np.fromstring(data,dtype=np.int16)[0::4]
            b = np.fromstring(data,dtype=np.int16)[1::4]
            c = np.fromstring(data,dtype=np.int16)[2::4]
            d = np.fromstring(data,dtype=np.int16)[3::4]
            frames0.append(a.tostring())
            frames1.append(b.tostring())
            frames2.append(c.tostring())
            frames3.append(d.tostring())
            stream.write(data)
        lb = int(input('Label: '))
        #lb = 0

        wf = wave.open(f'recordings/{today}/{i}.wav','wb')
        wf.setnchannels(1)
        wf.setsampwidth(p.get_sample_size(p.get_format_from_width(SAMPLE_WIDTH)))
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(b''.join(frames0))
        wf.close()

        label.append(lb)
        name.append(f'{i}.wav')        
        i += 1

        wf = wave.open(f'recordings/{today}/{i}.wav','wb')
        wf.setnchannels(1)
        wf.setsampwidth(p.get_sample_size(p.get_format_from_width(SAMPLE_WIDTH)))
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(b''.join(frames1))
        wf.close()

        label.append(lb)
        name.append(f'{i}.wav')        
        i += 1

        wf = wave.open(f'recordings/{today}/{i}.wav','wb')
        wf.setnchannels(1)
        wf.setsampwidth(p.get_sample_size(p.get_format_from_width(SAMPLE_WIDTH)))
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(b''.join(frames2))
        wf.close()

        label.append(lb)
        name.append(f'{i}.wav')        
        i += 1

        wf = wave.open(f'recordings/{today}/{i}.wav','wb')
        wf.setnchannels(1)
        wf.setsampwidth(p.get_sample_size(p.get_format_from_width(SAMPLE_WIDTH)))
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(b''.join(frames3))
        wf.close()

        label.append(lb)
        name.append(f'{i}.wav')        
        i += 1

        print("Done")
        stream.stop_stream()
        counter += 1
    except KeyboardInterrupt:
        df1 = {"name":name,"label":label}
        df1 = pd.DataFrame(df1)
        df1.to_csv('voice_data.csv',mode='a',index=False,header=False)
        print(df1)
        print('Exiting')
        break

if counter > 55:
    df1 = {"name":name,"label":label}
    df1 = pd.DataFrame(df1)
    df1.to_csv('voice_data.csv',mode='a',index=False,header=False)
    print(df1)
    print('Exiting')

stream.close()
p.terminate()











