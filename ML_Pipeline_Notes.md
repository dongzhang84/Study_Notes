# Machine Leanring Pipeline Notes



## 1. Fast ML 

### 1.1 Classification Quick Start

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



### 1.2 Some better methods

#### GridSeachCV

templete code based on Random Forest:

```python
from sklearn.model_selection import GridSearchCV

param_grid = { 
    'bootstrap': [True],
    'n_estimators': [100, 200, 300],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [3, 5, 10],
    'criterion' :['gini', 'entropy']
}

# Create a based model
rf = ensemble.RandomForestClassifier()

# Instantiate the grid search model
clf  = GridSearchCV(estimator = rf, 
                    cv=5, 
                    param_grid = param_grid)

clf.fit(X_train, y_train)
```

Evaluation:

```python
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
```



#### Class Weight for Imbalanced Classes

Consider binary classification, introduction see [this blog](https://towardsdatascience.com/practical-tips-for-class-imbalance-in-binary-classification-6ee29bcdb8a7):

```python
weight = 10

class_weights = {
0:1
1:weight
}

model = linear_model.LogisticRegression(random_state=None, max_iter=400, 
                                        solver='newton-cg', class_weight=class_weights)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
```

You can also apply **class_weight** to other algorithms, for example: **RandomForestClassifier**. 



#### Cross Valdiation

https://scikit-learn.org/stable/modules/cross_validation.html

![../_images/grid_search_cross_validation.png](https://scikit-learn.org/stable/_images/grid_search_cross_validation.png)



**Simple Way**



**K-Fold**

