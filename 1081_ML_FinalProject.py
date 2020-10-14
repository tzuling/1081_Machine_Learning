import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, Normalizer
from sklearn.ensemble import RandomForestClassifier

path = '/content/gdrive/My Drive/Colab Notebooks/SpotifyFeatures.csv'
data = pd.read_csv(path)

print("="*20 + "Descirbe Data" + "="*20)
for i in data.columns:
  print(data[i].describe())
print("\n")

print(data.keys())

print("="*20 + "Data Preprocessing" + "="*20)

print("1. data cleaning")
print(pd.isnull(data).sum())

print("2. Label Encoder")

print("2.1 Popularity(target) Distribution")
sns.distplot(data['popularity']).set_title('Popularity Distribution')


print("2.1.2 Popularity mapping")  # Popularity, Key, Mode, Time_Signature
# 根據四分位數將popularity分成四個等級


def popularity_mapping(popularity):
    if popularity <= 29:
        return 1
    elif popularity > 30 and popularity <= 43:
        return 2
    elif popularity > 43 and popularity <= 55:
        return 3
    elif popularity > 55:
        return 4


data['popularity'] = data['popularity'].map(popularity_mapping)

print("2.2 Genre, Key, Mode, Time_signature")
le = LabelEncoder()

print(data['genre'].unique())
data['genre'] = le.fit_transform(data['genre'])
print(data['genre'].unique())

# print(data['key'].unique())
# data['key'] = le.fit_transform(data['key'])
# print(data['key'].unique())

# data.loc[data["mode"] == 'Major', "mode"] = 1
# data.loc[data["mode"] == 'Minor', "mode"] = 0

print("2.3 duration_ms and round(large dtype('float32'), 2)")
# 將毫秒 -> 秒 (資料量很大，建議使用colab跑)
for i, ms in enumerate(data['duration_ms']):
    data['duration_ms'].loc[i] = round((ms * 0.001) / 60, 2)

print("3. Drop Data")
# 2019 TOP dataset 沒有以下欄位
data = data.drop(['track_id', 'track_name', 'artist_name',
                  'time_signature', 'instrumentalness', 'mode', 'key'], axis=1)

print("4. Normalization")
columns = ['tempo', 'energy', 'danceability', 'loudness',
           'liveness', 'valence', 'acousticness', 'speechiness']
norm_columns = data.loc[:, columns]
norm = Normalizer().fit(norm_columns)
norm_columns = norm.transform(norm_columns)
norm_columns = pd.DataFrame(norm_columns, columns=columns)
for i in norm_columns:
  data[i] = data[i].map(norm_columns[i])

# =============================================================================
X_columns = ['genre', 'tempo', 'energy', 'danceability', 'loudness',
             'liveness', 'valence', 'duration_ms', 'acousticness', 'speechiness']
y_column = ['popularity']

X = data.loc[:, X_columns]
y = data.loc[:, y_column]


print("5. Feature Importance Filter")
'''
利用 RandomForest 計算目標屬性與其他屬性之重要性
'''
rf = RandomForestClassifier()
rf.fit(X, y)
feature_importances = pd.DataFrame(rf.feature_importances_,
                                   index=X.columns,
                                   columns=['importance']).sort_values('importance', ascending=False)
print(feature_importances)
print("\n")

print("6. PCA")


# target
target = pd.read_csv("top50.csv", encoding="ISO-8859-1")

# print("="*20 + "Descirbe Data" + "="*20)
# print(target.describe())
# print("\n")

# print(target.keys())
