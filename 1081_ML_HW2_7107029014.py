import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, Normalizer, PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression

data = pd.read_csv("bank.csv", delimiter=",", header='infer')


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
data = data.drop(['contact', 'pdays', 'previous', 'poutcome'], axis=1)


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
# 結論：並未有 high correlation 之屬性(需大於0.5)

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
# 結論：每個屬性之importance都偏低，選取最低的三個欄位進行刪除: loan, housing, default

'''
4.3 Principal Component Analysis (PCA)
'''
print("=== Principal Component Analysis (PCA) ===")
pca = PCA(n_components=2)
pca.fit(X)
print(pca.explained_variance_ratio_) 
print(pca.explained_variance_)
print("\n")

'''
使用三種 Feature Selection 之方法，選取共同相對較低之重要性屬性進行刪除
->　loan, housing, default
'''
X_columns = ['age', 'job', 'marital', 'education', 'balance', 'month', 'duration', 'campaign']
# X_columns = ['balance', 'month', 'duration']
X = data.loc[:, X_columns]

# training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=100)

print("=== 顯示前10筆之資料 ===")
print(data.head(10))
print("\n")


# ==================== 5. Linear Regression method ====================
lr = LinearRegression()
lr.fit(X_train, y_train)
print("=== Linear Regression ===")
print(lr.coef_)
print(lr.score(X_test, y_test))  # 0.012999241500947
print("\n")


# ==================== 6. Ridge and Lasso Regression ====================
parameters = {'alpha': [20.0, 10.0, 5.0, 1.0, 0.1, 0.01, 0.005, 0.0025, 0.001, 0.00025]}

ridge = Ridge()
ridge_regressor = GridSearchCV(ridge, parameters)
ridge_regressor.fit(X, y)
print("=== Ridge Regression ===")
print(ridge_regressor.best_params_) # {'alpha': 20.0}
print(ridge_regressor.best_score_)  # 0.01950387384368864
print("\n")

lasso = Lasso()
lasso_regressor = GridSearchCV(lasso, parameters)
lasso_regressor.fit(X, y)
print("=== Lasso Regression ===")
print(lasso_regressor.best_params_)  # {'alpha': 0.001}
print(lasso_regressor.best_score_)  # 0.020125359914734183
print("\n")


'''
跑完以上三種Linear Regression之方法，結論為：資料可能並非線性資料，因此下面試跑兩種非線性之方法
'''
# ==================== 7. Polynomial Regression ====================
print("=== Polynomial Regression ===")
polynomial_features = PolynomialFeatures(degree=4)
'''
degree = 4:
rmse = 0.29089971448654184
r2 = 0.17177852403720895

degree = 5:
rmse = 0.2582038969862041
rmse = 0.347492537428428
-> 可能有過度學習之現象，因此選擇 degree = 4
'''
x_poly = polynomial_features.fit_transform(X)

model = LinearRegression()
model.fit(x_poly, y)
y_poly_pred = model.predict(x_poly)

rmse = np.sqrt(mean_squared_error(y, y_poly_pred))
r2 = r2_score(y, y_poly_pred)
print(rmse)  # 0.29089971448654184
print(r2)  # 0.17177852403720895
print("\n")


# ==================== 8. Logistic Regression ====================
lg = LogisticRegression()
lg.fit(X_train, y_train)
print("=== Logistic Regression ===")
print(lg.coef_)
print(lg.score(X_test, y_test))  # 0.8902627511591963
print("\n")

