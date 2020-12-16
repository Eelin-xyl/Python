import pandas as pd

dataset = pd.read_csv(r'Python\\DataAnalysis\\SpotifyFeatures(cleaned3).csv',encoding='gbk')

for i, v in enumerate(dataset.key):
    if v == 'C':
        dataset.key[i] = 1
    elif v == 'C#':
        dataset.key[i] = 1.5
    elif v == 'D':
        dataset.key[i] = 2
    elif v == 'D#':
        dataset.key[i] = 2.5
    elif v == 'E':
        dataset.key[i] = 3
    elif v == 'F':
        dataset.key[i] = 4
    elif v == 'F#':
        dataset.key[i] = 4.5
    elif v == 'G':
        dataset.key[i] = 5
    elif v == 'G#':
        dataset.key[i] = 5.5
    elif v == 'A':
        dataset.key[i] = 6
    elif v == 'A#':
        dataset.key[i] = 6.5
    elif v == 'B':
        dataset.key[i] = 7
    

dataset.to_csv(r'Python\DataAnalysis\SpotifyFeatures(key).csv', encoding = 'gbk')