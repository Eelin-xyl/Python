import pandas as pd

data = pd.read_csv(r'Python\DataAnalysis\SpotifyFeatures(cleaned2).csv', encoding="gbk",)
data = data.iloc[:100]
print(data.head(6))
data.to_csv(r'Python\DataAnalysis\SpotifyFeatures(cleaned3).csv')