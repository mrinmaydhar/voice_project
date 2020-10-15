import pandas as pd

df = pd.read_csv('voice_data_testing.csv')
name = ['1.wav','2.wav','3.wav','4.wav']
label = [1,1,1,1]
df['name'] = name
df['label'] = label
df.to_csv('voice_data_testing.csv',index=False)