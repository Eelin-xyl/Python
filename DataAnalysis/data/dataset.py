import pandas as pd
import numpy as np
import random

dataset = pd.read_csv(r'Python\\DataAnalysis\\SpotifyFeatures(cleaned2).csv',encoding='gbk')
lst = [[[] for k in range(11)] for l in range(25)]

for i in range(223253):
    if dataset.iloc[i,0] == 'Alternative':
        for j in range(11):
            t = float(dataset.iloc[i,j+2])
            lst[0][j].append(t)

    elif dataset.iloc[i,0] == 'Anime':
        for j in range(11):
            t = float(dataset.iloc[i,j+2])
            lst[1][j].append(t)

    elif dataset.iloc[i,0] == 'Blues':
        for j in range(11):
            t = float(dataset.iloc[i,j+2])
            lst[2][j].append(t)
            
    elif dataset.iloc[i,0] == "Children's Music":
        for j in range(11):
            t = float(dataset.iloc[i,j+2])
            lst[3][j].append(t)
            
    elif dataset.iloc[i,0] == 'Classical':
        for j in range(11):
            t = float(dataset.iloc[i,j+2])
            lst[4][j].append(t)
            
    elif dataset.iloc[i,0] == 'Comedy':
        for j in range(11):
            t = float(dataset.iloc[i,j+2])
            lst[5][j].append(t)
            
    elif dataset.iloc[i,0] == 'Country ':
        for j in range(11):
            t = float(dataset.iloc[i,j+2])
            lst[6][j].append(t)
            
    elif dataset.iloc[i,0] == 'Dance':
        for j in range(11):
            t = float(dataset.iloc[i,j+2])
            lst[7][j].append(t)
            
    elif dataset.iloc[i,0] == 'Electronic':
        for j in range(11):
            t = float(dataset.iloc[i,j+2])
            lst[8][j].append(t)
            
    elif dataset.iloc[i,0] == 'Folk':
        for j in range(11):
            t = float(dataset.iloc[i,j+2])
            lst[9][j].append(t)
            
    elif dataset.iloc[i,0] == 'Hip-Hop':
        for j in range(11):
            t = float(dataset.iloc[i,j+2])
            lst[10][j].append(t)
            
    elif dataset.iloc[i,0] == 'Indie':
        for j in range(11):
            t = float(dataset.iloc[i,j+2])
            lst[11][j].append(t)
            
    elif dataset.iloc[i,0] == 'Jazz':
        for j in range(11):
            t = float(dataset.iloc[i,j+2])
            lst[12][j].append(t)
            
    elif dataset.iloc[i,0] == 'Movie':
        for j in range(11):
            t = float(dataset.iloc[i,j+2])
            lst[13][j].append(t)
            
    elif dataset.iloc[i,0] == 'Opera':
        for j in range(11):
            t = float(dataset.iloc[i,j+2])
            lst[14][j].append(t)
            
    elif dataset.iloc[i,0] == 'Pop':
        for j in range(11):
            t = float(dataset.iloc[i,j+2])
            lst[15][j].append(t)
            
    elif dataset.iloc[i,0] == 'R&B':
        for j in range(11):
            t = float(dataset.iloc[i,j+2])
            lst[16][j].append(t)
            
    elif dataset.iloc[i,0] == 'Rap':
        for j in range(11):
            t = float(dataset.iloc[i,j+2])
            lst[17][j].append(t)
            
    elif dataset.iloc[i,0] == 'Reggae':
        for j in range(11):
            t = float(dataset.iloc[i,j+2])
            lst[18][j].append(t)
            
    elif dataset.iloc[i,0] == 'Reggaeton':
        for j in range(11):
            t = float(dataset.iloc[i,j+2])
            lst[19][j].append(t)
            
    elif dataset.iloc[i,0] == 'Rock':
        for j in range(11):
            t = float(dataset.iloc[i,j+2])
            lst[20][j].append(t)
            
    elif dataset.iloc[i,0] == 'Ska':
        for j in range(11):
            t = float(dataset.iloc[i,j+2])
            lst[21][j].append(t)
            
    elif dataset.iloc[i,0] == 'Soul':
        for j in range(11):
            t = float(dataset.iloc[i,j+2])
            lst[22][j].append(t)
            
    elif dataset.iloc[i,0] == 'Soundtrack':
        for j in range(11):
            t = float(dataset.iloc[i,j+2])
            lst[23][j].append(t)
            
    elif dataset.iloc[i,0] == 'World':
        for j in range(11):
            t = float(dataset.iloc[i,j+2])
            lst[24][j].append(t)

for i in range(len(lst)):
    for j, w in enumerate(lst[i]):
        a = round(np.mean(w), 4)
        b = round(np.std(w,ddof=1), 4)
        lst[i][j] = [a-b, a+b]

for i in range(223253):
    if dataset.iloc[i,0] == 'Alternative':
        for j in range(11):
            dataset.iloc[i,j+2] = round(random.uniform(lst[0][j][0], lst[0][j][1]),4)

    elif dataset.iloc[i,0] == 'Anime':
        for j in range(11):
            dataset.iloc[i,j+2] = round(random.uniform(lst[1][j][0], lst[1][j][1]),4)

    elif dataset.iloc[i,0] == 'Blues':
        for j in range(11):
            dataset.iloc[i,j+2] = round(random.uniform(lst[2][j][0], lst[2][j][1]),4)

    elif dataset.iloc[i,0] == "Children's Music":
        for j in range(11):
            dataset.iloc[i,j+2] = round(random.uniform(lst[3][j][0], lst[3][j][1]),4)

    elif dataset.iloc[i,0] == 'Classical':
        for j in range(11):
            dataset.iloc[i,j+2] = round(random.uniform(lst[4][j][0], lst[4][j][1]),4)

    elif dataset.iloc[i,0] == 'Comedy':
        for j in range(11):
            dataset.iloc[i,j+2] = round(random.uniform(lst[5][j][0], lst[5][j][1]),4)

    elif dataset.iloc[i,0] == 'Country ':
        for j in range(11):
            dataset.iloc[i,j+2] = round(random.uniform(lst[6][j][0], lst[6][j][1]),4)
 
    elif dataset.iloc[i,0] == 'Dance':
        for j in range(11):
            dataset.iloc[i,j+2] = round(random.uniform(lst[7][j][0], lst[7][j][1]),4)
 
    elif dataset.iloc[i,0] == 'Electronic':
        for j in range(11):
            dataset.iloc[i,j+2] = round(random.uniform(lst[8][j][0], lst[8][j][1]),4)

    elif dataset.iloc[i,0] == 'Folk':
        for j in range(11):
            dataset.iloc[i,j+2] = round(random.uniform(lst[9][j][0], lst[9][j][1]),4)

    elif dataset.iloc[i,0] == 'Hip-Hop':
        for j in range(11):
            dataset.iloc[i,j+2] = round(random.uniform(lst[10][j][0], lst[10][j][1]),4)

    elif dataset.iloc[i,0] == 'Indie':
        for j in range(11):
            dataset.iloc[i,j+2] = round(random.uniform(lst[11][j][0], lst[11][j][1]),4)

    elif dataset.iloc[i,0] == 'Jazz':
        for j in range(11):
            dataset.iloc[i,j+2] = round(random.uniform(lst[12][j][0], lst[12][j][1]),4)

    elif dataset.iloc[i,0] == 'Movie':
        for j in range(11):
            dataset.iloc[i,j+2] = round(random.uniform(lst[13][j][0], lst[13][j][1]),4)

    elif dataset.iloc[i,0] == 'Opera':
        for j in range(11):
            dataset.iloc[i,j+2] = round(random.uniform(lst[14][j][0], lst[14][j][1]),4)

    elif dataset.iloc[i,0] == 'Pop':
        for j in range(11):
            dataset.iloc[i,j+2] = round(random.uniform(lst[15][j][0], lst[15][j][1]),4)

    elif dataset.iloc[i,0] == 'R&B':
        for j in range(11):
            dataset.iloc[i,j+2] = round(random.uniform(lst[16][j][0], lst[16][j][1]),4)

    elif dataset.iloc[i,0] == 'Rap':
        for j in range(11):
            dataset.iloc[i,j+2] = round(random.uniform(lst[17][j][0], lst[17][j][1]),4)

    elif dataset.iloc[i,0] == 'Reggae':
        for j in range(11):
            dataset.iloc[i,j+2] = round(random.uniform(lst[18][j][0], lst[18][j][1]),4)

    elif dataset.iloc[i,0] == 'Reggaeton':
        for j in range(11):
            dataset.iloc[i,j+2] = round(random.uniform(lst[19][j][0], lst[19][j][1]),4)

    elif dataset.iloc[i,0] == 'Rock':
        for j in range(11):
            dataset.iloc[i,j+2] = round(random.uniform(lst[20][j][0], lst[20][j][1]),4)

    elif dataset.iloc[i,0] == 'Ska':
        for j in range(11):
            dataset.iloc[i,j+2] = round(random.uniform(lst[21][j][0], lst[12][j][1]),4)

    elif dataset.iloc[i,0] == 'Soul':
        for j in range(11):
            dataset.iloc[i,j+2] = round(random.uniform(lst[22][j][0], lst[22][j][1]),4)
 
    elif dataset.iloc[i,0] == 'Soundtrack':
        for j in range(11):
            dataset.iloc[i,j+2] = round(random.uniform(lst[23][j][0], lst[23][j][1]),4)

    elif dataset.iloc[i,0] == 'World':
        for j in range(11):
            dataset.iloc[i,j+2] = round(random.uniform(lst[24][j][0], lst[24][j][1]),4)

dataset.to_csv(r'Python\DataAnalysis\SpotifyFeatures(cleaned3).csv', encoding = 'gbk')
print('YES')