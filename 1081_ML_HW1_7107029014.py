import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, LabelBinarizer, Imputer

train = pd.read_csv("./Train.csv")
train.shape

pd.set_option('display.max_columns', None)

# 移除ID性質的feature
train = train.drop(['Item_Identifier', 'Outlet_Identifier'], axis=1)

# 套件自行決定label encoder -> 結果為一維陣列
le = LabelEncoder() 

# 1. Item_Fat_Content data cleaning
'''
['Low Fat' 'Regular' 'low fat' 'LF' 'reg']

data cleaning steps:
raw data -> replace -> label encoder
Low Fat, LF, low fat -> Low Fat -> 0
Regular, reg -> Regular -> 1
'''
    # raw data
category = train['Item_Fat_Content']

    # data cleaning
category = [x.replace('LF', 'Low Fat') for x in category]
category = [x.replace('low fat', 'Low Fat') for x in category]
category = [x.replace('reg', 'Regular') for x in category]

    # category = category.replace(regex=r'^reg', value='regular') 
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.replace.html

train['Item_Fat_Content'] = category

    # label encoder
train['Item_Fat_Content'] = le.fit_transform(train['Item_Fat_Content'])

# 2. Item_Type data cleaning
'''
{'Dairy': 4, 'Soft Drinks': 14, 'Meat': 10, 'Fruits and Vegetables': 6, 
'Household': 9, 'Baking Goods': 0, 'Snack Foods': 13, 'Frozen Foods': 5, 
'Breakfast': 2, 'Health and Hygiene': 8, 'Hard Drinks': 7, 'Canned': 3, 
'Breads': 1, 'Starchy Foods': 15, 'Others': 11, 'Seafood': 12}
'''
train['Item_Type'] = le.fit_transform(train['Item_Type'])

'''
# Item_Type label encoder https://www.lfhacks.com/tech/python-list-element-replace
dic = {}
    # 欄位
Item_Type = train['Item_Type'].unique()
    # label encoder
Item_Type_encoding = le.fit_transform(Item_Type)

    # dictionary {欄位:label encoder}
for i in range(len(Item_Type)):
    dic.update({Item_Type[i]:Item_Type_encoding[i]})

category = [dic[x] if x in dic else x for x in train['Item_Type']]
train['Item_Type'] = np.asarray(category)
'''

# 3. Item_Weight, Outlet_Size data cleaning

    # replace the missing value to median
    # train['Item_Weight'] = train['Item_Weight'].fillna(train['Item_Weight'].median(), inplace=True)
    # NaN -> median() -> 12.857645
imp = Imputer(strategy="mean")
Item_Weight = imp.fit_transform(train['Item_Weight'].values.reshape(-1, 1))
train['Item_Weight'] = Item_Weight

    # replace the missing value to most frequent
    # NaN -> most frequent -> Medium
'''
High -> 0
Medium -> 1
Small -> 2
'''
train['Outlet_Size'].fillna(value=train['Outlet_Size'].mode()[0], inplace=True)
train['Outlet_Size'] = le.fit_transform(train['Outlet_Size'])

# 4. other categories
# Outlet_Location_Type label encoder
'''
Tier 1 -> 0
Tier 2 -> 1
Tier 3 -> 2
'''
train['Outlet_Location_Type'] = le.fit_transform(train['Outlet_Location_Type'])

# Outlet_Type label encoder
'''
Grocery Store -> 0
Supermarket Type1 -> 1
Supermarket Type2 -> 2
Supermarket Type3 -> 3
'''
train['Outlet_Type'] = le.fit_transform(train['Outlet_Type'])

# whole train data head(10)
print(train.head(10))
