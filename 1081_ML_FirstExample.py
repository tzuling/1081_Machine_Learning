import pandas as pd


data = pd.read_csv("bank.csv", delimiter=";", header='infer')
data.y.replace(('yes', 'no'), (1, 0), inplace=True)
data.loan.replace(('yes', 'no'), (1, 0), inplace=True)

# final = data.drop(['job', 'marital', 'education', 'default', 'housing', 'contact',
#                    'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome'], axis=1)

# X = final.drop(['y'], axis=1)
# y = final.drop(['age', 'balance', 'loan'], axis=1)

X = data.loc[:, ['age', 'balance', 'loan']]
y = data.loc[:, ['y']]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=100)

# from sklearn.linear_model import LinearRegression
# lr = LinearRegression()
# lr.fit(X_train, y_train)
# lr.score(X_test, y_test)
# print(lr.score(X_test, y_test))
# print(lr.coef_)

from sklearn import tree
dt1 = tree.DecisionTreeClassifier(max_depth=4)
dt1.fit(X_train, y_train)
dt1.score(X_test, y_test)
print(dt1.score(X_test, y_test))

# 請考慮其他的 features，並重新完成分類
# 請重新思考 age，在處理的時候，我們要不要對 age 進行前處理，例如 30-40, 41-50, …
