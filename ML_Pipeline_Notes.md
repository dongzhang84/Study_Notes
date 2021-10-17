# Machine Leanring Pipeline Notes



## 1. Classification Quick Start

If there is no missing data, and all features are number and ready, deploy the following steps:

Load libaries

```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler
```

### 1.1. Split the data

```python
train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df.Class)
```

Note that I use **df** without splitting it to train and label. Also I use **stratify** for imblanced data split. 

A more commonlly used way

```python
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
```



### 1.2. Scale the data

This is not necessary, but you may want to do it:

```python
#from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range = (-1, 1))
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
```

Another better way to do it:

```python
df_scaled = df.copy()
scaler = MinMaxScaler()
df_scaled[feature_list] = scaler.fit_transform(df[feature_list])
df_scaled.head()
```

And there are also other scalers, for example **StandardScaler**:

```python
df_scaled = df.copy()
scaler = StandardScaler()
df_scaled[feature_list] = scaler.fit_transform(df[feature_list])
```

or **RobustScaler**:

```python
df_scaled = df.copy()
scaler = RobustScaler()
df_scaled[feature_list] = scaler.fit_transform(df[feature_list])
```



### 1.3. Encode Categorical Variables

```python
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
encoder.fit(df[feature])
df['feature_encode'] = encoder.transform(df['feature'])
```



### 1.4. Some Baseline Models

**Logistic Regression**

```python
from sklearn import linear_model

model = linear_model.LogisticRegression()
model = linear_model.LogisticRegressionCV()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
```

However, sometimes you do not know y_test. So you should use **Cross Validation** for validation dataset. 

Here is the demonstration of **k-fold cross validation**:

![img](https://scikit-learn.org/stable/_images/grid_search_cross_validation.png)



Example code (see [this link](from sklearn.model_selection import cross_val_score)):

```python
from sklearn.model_selection import cross_val_score

model = linear_model.LogisticRegression()
scores = cross_val_score(model, X_train, y_train, scoring = 'f1', cv=5)
scores.mean(), scores.std()
```

All cross_val_score see [this link](https://scikit-learn.org/stable/modules/model_evaluation.html). 



**Random Forest**

```python
from sklearn import ensemble
from sklearn.model_selection import cross_val_score

model = ensemble.RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
```

```
model = ensemble.RandomForestClassifier()
scores = cross_val_score(model, X_train, y_train, scoring = 'f1', cv=5)
scores.mean(), scores.std()
```



**XGBoost**

```python
# XGBoost looks good

import xgboost
from sklearn.model_selection import cross_val_score

model = xgboost.XGBClassifier()  
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# or 

scores = cross_val_score(model, X_train, y_train, scoring = 'f1', cv=5)
scores.mean(), scores.std()
```



**LightGBM**

```python
# lightgbm pretty bad

import lightgbm

model = lightgbm.LGBMClassifier()  
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# or 

scores = cross_val_score(model, X_train, y_train, scoring = 'f1', cv=5)
scores.mean(), scores.std()
```



More algorithms please see the [Titantic templete](https://github.com/dongzhang84/data_challenges/blob/master/Titanic.ipynb).

These algorithms can help you build a baseline model very fast. 



### 1.5. Some better methods

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

Find the best parameters:

```python
print(grid.best_estimator_)
print(grid.best_params_)
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





## 2. Regression Quick Start



### 2.1 Metrics

Here is a good metric template for Root Mean Square Deviation: 

```python
#Validation function: RMSE

n_folds = 5

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(X_train.values)
    rmse= np.sqrt(-cross_val_score(model, X_train.values, y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)
```



### 2.2. Baseline Models

**Lasso Regression**

```python
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error

lasso = make_pipeline(RobustScaler(), Lasso(alpha = alpha_test, random_state=1))
lasso.fit(X_train, y_train)
y_pred = lasso.predict(X_test)
print(np.sqrt(mean_squared_error(y_test, y_pred))
```

or one can do:

```python
from sklearn.model_selection import cross_val_score

model = lasso
np.sqrt(-cross_val_score(model, X_train, y_train, scoring="neg_mean_squared_error", cv = 5))
```



**Elastic Net**

```python
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error

ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=alpha_test, l1_ratio=.8, random_state=3))
ENet.fit(X_train, y_train)
y_pred = ENet.predict(X_test)
print(np.sqrt(mean_squared_error(y_test, y_pred))
```

or

```python
from sklearn.model_selection import cross_val_score

model = ENet
np.sqrt(-cross_val_score(model, X_train, y_train, scoring="neg_mean_squared_error", cv = 5))
```



**LightGBM Regressor**

```python
import lightgbm as lgb

model_lgb = lgb.LGBMRegressor(learning_rate = 0.1, n_estimators = 300, objective='regression')

model_lgb.fit(X_train, y_train)
y_pred = model_lgb.predict(X_test)
print(np.sqrt(mean_squared_error(y_test, y_pred))
      
# or

np.sqrt(-cross_val_score(model_lgb, X_train, y_train, 
                         scoring="neg_mean_squared_error", cv = 5))
```



**XGBoost Regressor**

```python
import xgboos as xgb

model_xgb = xgb.XGBRegressor()

model_xgb.fit(X_train, y_train)
y_pred = model_xgb.predict(X_test)
print(np.sqrt(mean_squared_error(y_test, y_pred))
      
# or

np.sqrt(-cross_val_score(model_xgb, X_train, y_train, 
                         scoring="neg_mean_squared_error", cv = 5))
```







### 3. Feature Engineering

How to encode Categorical Variables see **Section 1.3**. 



#### 3.1. Automatic Feature Engineering

See [this note](https://github.com/dongzhang84/Study_Notes/blob/main/FeatureTools_Notes.md). 





