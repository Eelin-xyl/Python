import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Python\\DataAnalysis\\SpotifyFeatures2.csv')

list = ['artist_name', 'track_name', 'key', 'popularity', 'acousticness', 
'danceability',	'duration_ms', 'energy', 'instrumentalness', 'liveness', 
'loudness', 'mode', 'speechiness', 'tempo', 'time_signature', 'valence',]
for i in list:
    print(dataset[i].describe(include='all'))