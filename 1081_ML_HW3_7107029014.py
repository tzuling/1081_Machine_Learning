import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, Normalizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB 
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR,SVC 

path = '/content/gdrive/My Drive/Colab Notebooks/bank.csv'
data = pd.read_csv(path)

# ==================== 1. 處理有遺失值之欄位 ====================
# 以下數字為個別欄位之遺失值數量
'''
job -> unknown: 38/4521
education -> unknown: 187/4521
'''
data = data[data.job != 'unknown']
data = data[data.education != 'unknown']

# 以下為遺失值過多之欄位，直接移除
'''
contact -> unknown: 1324/4521
pdays -> -1: 3705/4521
previous -> 0: 3705/4521
poutcome -> unknown: 3705/4521
'''
# data = data.drop(['contact', 'pdays', 'previous', 'poutcome'], axis=1)


# ==================== 2. 將類別型資料轉成數字型態 ====================
le = LabelEncoder()

columns = data.columns.values  # DataFrame to Array
for i in columns:
    data[i] = le.fit_transform(data[i])


# ==================== 3. normalization (balance, duration) ====================
'''
數值型態都應進行 normalize
'''
norm_columns = data.loc[:, ['balance', 'duration']]

norm = Normalizer().fit(norm_columns)
norm_columns = norm.transform(norm_columns)
norm_columns = pd.DataFrame(norm_columns, columns=['balance', 'duration'])
# norm = Normalizer().fit(data.loc[:, ['balance']])
# data.loc[:, ['balance']] = norm.transform(norm)

data['balance'] = data['balance'].map(norm_columns['balance'])
data['duration'] = data['duration'].map(norm_columns['duration'])


# 3.1 age, campaign 將 value -> interval 
'''
類別型資料不應做normalize，但又有過多value，因此使用 mapping 將 value 分類成 interval
'''
def age_mapping(age):
    if age <= 22:  # student
        return 0
    elif age > 22 and age <= 30:
        return 1
    elif age > 30 and age <= 40:
        return 2
    elif age > 40 and age <= 50:
        return 3
    elif age > 50 and age <= 60:
        return 4
    elif age > 60:  # elder
        return 5

def campaign_mapping(campaign):
    if campaign == 0:
        return 0
    elif campaign == 1:
        return 1
    elif campaign == 2:
        return 2
    elif campaign == 3:
        return 3
    elif campaign == 4:
        return 4
    elif campaign >= 5: # 5以上已為少數，將少數的資料合為一類別，以減少分類
        return 5

data['age'] = data['age'].map(age_mapping)
data['campaign'] = data['campaign'].map(campaign_mapping)


# ==================== 4. Feature Selection ====================
'''
先將 day 欄位刪除，時間型態資料留下意義不大，且會因為 value 很多影響其準確性。
僅留 month 欄位即可
'''
X_columns = ['age', 'job', 'marital', 'education', 'default',
             'balance', 'housing', 'loan', 'month', 'duration', 'campaign']
y_column = ['y']

X = data.loc[:, X_columns]
y = data.loc[:, y_column]

'''
4.1 High Correlation Filter: 
利用 coef 計算目標屬性與其他屬性之重要性
'''
print("=== High Correlation Filter ===")
print(data.corr())
print("\n")
# 結論：刪除負coef值：default, balance, housing, loan, day, month

'''
4.2 Feature Importance Filter:
利用 RandomForest 計算目標屬性與其他屬性之重要性
'''
print("=== Feature Importance Filter ===")
rf = RandomForestClassifier()
rf.fit(X, y)
feature_importances = pd.DataFrame(rf.feature_importances_,
                                   index=X.columns,
                                   columns=['importance']).sort_values('importance', ascending=False)

print(feature_importances)
print("\n")
# 結論：每個屬性之importance都偏低，選取最低的三個欄位進行刪除: loan, housing, default, campaign

'''
4.3 Principal Component Analysis (PCA)
'''
print("=== Principal Component Analysis (PCA) ===")
pca = PCA(n_components = 'mle')
pca.fit(X)
print(pca.explained_variance_ratio_) 
print(pca.explained_variance_)
print("\n")

'''
使用三種 Feature Selection 之方法，選取共同相對較低之重要性屬性進行刪除
->　loan, housing, default
'''
X_columns = ['age', 'job', 'marital', 'education', 'duration']
X = data.loc[:, X_columns]

print("=== 顯示前10筆之資料 ===")
print(data.head(10))
print("\n")


# training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=100)
	
# ==================== Naive Bayes ============================
clf = GaussianNB()
clf.fit(X_train, y_train)
bayes_pred = clf.predict(X_test)
print("=== Naive Bayes ===")
print(bayes_pred)

acc = accuracy_score(y_test, bayes_pred)
print("accuracy: ", acc)
print("\n")

# ==================== Logistic Regression ====================
lg = LogisticRegression()
lg.fit(X_train, y_train)
print("=== Logistic Regression ===")
print(lg.coef_)
print(lg.score(X_test, y_test))  # 0.8902627511591963
print("\n")

# ==================== SVM ====================================
param_grid = {'C': [1, 5, 10, 15, 20],
              'gamma': [0.1, 0.25, 0.5, 1]}
clf = GridSearchCV(SVC(kernel='rbf'), param_grid)
clf.fit(X_train, y_train)

print("找出較好的C及gamma")
best_C = clf.best_estimator_.C
best_gamma = clf.best_estimator_.gamma

print('C =', best_C) # C=20
print('gamma =', best_gamma) # gamma=0.1

svm_rbf = SVC(kernel='rbf', gamma=best_gamma, C=best_C)
svm_rbf.fit(X_train, y_train)
print("分類機器人參數")
print(svm_rbf)

#評估新參數的預測結果好壞
print("評估新參數的預測結果好壞")
predictions = svm_rbf.predict(X_test)
grid_predictions = clf.predict(X_test)
print('\n')
print(confusion_matrix(y_test,predictions))
print('\n')
print(classification_report(y_test,grid_predictions))

'''
              precision    recall  f1-score   support

           0       0.89      1.00      0.94      1152
           1       0.17      0.01      0.01       142

    accuracy                           0.89      1294
   macro avg       0.53      0.50      0.48      1294
weighted avg       0.81      0.89      0.84      1294
'''
