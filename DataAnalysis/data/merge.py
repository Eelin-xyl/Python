import pandas as pd

dataset = pd.read_csv(r'Python\\DataAnalysis\\SpotifyFeatures(cleaned).csv',encoding='gbk')

for i, v in enumerate(dataset.genre):
    if v == 'R&B' or v == 'Country' or v == 'Blues' or v == 'Hip-Hop' or v == 'Rap' or v == 'Reggae' or v == 'Reggaeton' or v == 'Jazz' or v == 'Rock' or v == 'Ska' or v == 'Soul':
        dataset.genre[i] = 1
    elif v == 'Movie' or v == 'Anime' or v == 'Opera' or v == "Children's Music" or v == "Children’s Music" or v == 'Comedy' or v == 'Soundtrack':
        dataset.genre[i] = 2
    elif v == 'A Capella' or v == 'Alternative' or v == 'Indie':
        dataset.genre[i] = 3
    elif v == 'Dance' or v == 'Electronic' or v == 'Pop':
        dataset.genre[i] = 4
    elif v == 'Folk' or v == 'Classical' or v == 'World':
        dataset.genre[i] = 5

dataset.to_csv(r'Python\DataAnalysis\SpotifyFeatures(merge).csv', encoding = 'gbk')