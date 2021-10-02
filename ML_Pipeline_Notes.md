# Machine Leanring Pipeline Notes



## Fast ML 

### Classification

If there is no missing data, and all features are number and ready, deploy the following steps:

Load libaries

```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler
```

**Split the data**

```python
train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df.Class)
```

Note that I use **df** without splitting it to train and label. Also I use **stratify** for imblanced data split. 

A more commonlly used way

```python
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
```



**Scale the data**

This is not necessary, but you may want to do it:

```python
#from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range = (-1, 1))
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
```



**Logistic Regression**

```python
from sklearn import linear_model

model = linear_model.LogisticRegression()
model = linear_model.LogisticRegressionCV()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
```



**Random Forest**

```python
from sklearn import ensemble

model = ensemble.RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
```



**XGBoost**

```python
# XGBoost looks good

import xgboost

model = xgboost.XGBClassifier()  
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
```



**LightGBM**

```python
# lightgbm pretty bad

import lightgbm

model = lightgbm.LGBMClassifier()  
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
```



More algorithms please see the [Titantic templete](https://github.com/dongzhang84/data_challenges/blob/master/Titanic.ipynb).

These algorithms can help you build a baseline model very fast. 