# Machine Leanring Pipeline Notes

Content

1. Classification Quick Start
   - Split the data (non-cv treatment)
   - Scale the data
   - Encode Categorical Data (more feature engineering see Section 3)
   - Fill null/missing values
   - Feature Engineering (see below)
   - Some baseline Models (Logistic Regression, Random Forest, XGBoost, LightGBM)
   - Some advanced methods (GridSearchCV, Imbalanced Data)
2. Regression Quick Start
   - Metrics
   - Baseline Models (Lasso Regression, Elastic Net, LightGBM, XGBoost regressor)
3. Feature Engineering
   - Encode, One-hot, Featuretools
   - Feature selection (remove collinear features, Lasso feature selection, etc.)
   - Rank Feature Importance (LightGBM and Lasso template)
4. Deep Learning Basic and Quick Start
   - Keras Quick Start







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

Convert a variable to categorical data:

```python
df.feature.astype(str)
```

Use **Label Encoder**:

```python
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
encoder.fit(df[feature])
df['feature_encode'] = encoder.transform(df['feature'])
```

More feature engineering see Section 3. 



### 1.4. Fill Null Values

Check missing data:

```
df.isnull.sum()
```

 Some methods to drop null values:

```python
# drop any null values:

df.dropna()

# drop null values for a particular feature/column

df[df.feature.notull()]
```

Fill null with median value:

```python
df[feature]fillna(df[feature]median(), inplace=True)
```

Do it using  **apply function**:

```python
df.apply(lambda x: x.fillna(x.median()), axis=0)
```





### 1.5. Some Baseline Models

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



### 1.6. Some advanced methods

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
0:1,
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

A  good reference can be found [here](https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard). 

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



#### 3.1. Encode Categorical Variable

How to encode Categorical Variables see **Section 1.3**. Another way to do encode is to use **One-Hot**. 

```python
import pandas as pd

df_dummy = pd.get_dummies(df)
```



#### 3.1. Automatic Feature Engineering

See [this note](https://github.com/dongzhang84/Study_Notes/blob/main/FeatureTools_Notes.md). Featuretools give a good way to do feature engineering automatically. 



#### 3.2 Feature Selection



**Remove Collinearity**

Check correlation matrix:

```python
#correlation matrix
import seaborn as sns

corrmat = df.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.9, square=True);
```

Remove collinear features template:

```python
# Threshold for removing correlated variables, I set threshold to 0.9.

threshold = 0.9

# Absolute value correlation matrix
corr_matrix = df.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# correlation function visualization, not necessary
upper.style.applymap(highlight)

# Select collinear features above threshold
collinear_features = [column for column in upper.columns if any(upper[column] > threshold)]
df_filtered = df.drop(columns = collinear_features)

print('The number of features that passed the collinearity threshold: ', df_filtered.shape[1])
```



**Feature Selection by Lasso**

```python
from sklearn.feature_selection import SelectFromModel

lasso = LassoCV(alphas=[1.e-5, 0.001, 0.1], cv=5).fit(X_train, y_train)
model = SelectFromModel(lasso, prefit=True)
X_new = model.transform(X_train)
X_selected_df = pd.DataFrame(X_new, columns=[X_train.columns[i] for i in range(len(X_train.columns)) 
                                             if model.get_support()[i]])
```





Some other feature selection methods see [this link](https://github.com/dongzhang84/Featuretools/blob/main/Titanic_Featuretools.ipynb). 

- By LinearSVC
- by the SelectKBest with Chi-2
- by the Recursive Feature Elimination (RFE) with Logistic Regression
- by the Recursive Feature Elimination (RFE) with Random Forest



#### 3.3 Check Feature Importance rankings

For LightGBM:

```python
# check feature importance, top 20 importance

fig, ax = plt.subplots(figsize=(10, 6))
lgb.plot_importance(model_lgb, max_num_features=20, ax=ax)
plt.show()
```

For XGBoost:

```python
# check feature importance, top 20 importance

fig, ax = plt.subplots(figsize=(10, 6))
xgb.plot_importance(model_xgb, max_num_features=20, ax=ax)
plt.show()
```



For Lasso:

```python
model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005]).fit(X_train, y_train)
coef = pd.Series(model_lasso.coef_, index = X_train.columns)
```





### 4. Deep Learning Basic and Quick Start

Use Keras, [reference](https://www.kaggle.com/hugosjoberg/house-prices-prediction-using-keras):

```python
def create_model():
    # create model
    model = Sequential()
    model.add(Dense(10, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(40, activation='relu'))
    model.add(Dense(1))
    # Compile model
    model.compile(optimizer ='adam', loss = 'mean_squared_error', 
              metrics =[metrics.mae])
    return model
```

```python
model = create_model()
model.summary()
```

One can see: 

```python
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense (Dense)                (None, 10)                1150      
_________________________________________________________________
dense_1 (Dense)              (None, 30)                330       
_________________________________________________________________
dense_2 (Dense)              (None, 40)                1240      
_________________________________________________________________
dense_3 (Dense)              (None, 1)                 41        
=================================================================
Total params: 2,761
Trainable params: 2,761
Non-trainable params: 0
```

```python
history = model.fit(X_train, y_train, epochs=50, batch_size=100)
```

