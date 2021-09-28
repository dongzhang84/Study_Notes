# Feature Engineering and Machine Learning using FeatureTools



First one needs a clear data. Take the Titanic problem for eaxmple, in [this notebook](https://github.com/dongzhang84/Featuretools/blob/main/Titanic_Featuretools%20automation_1.ipynb) I split the original training dataset to train and test again. 

The train and test data needs to be clear, which means initial FE should be applied to remove null data (needs more test if this step is necessary). 

The work is based on the two notebooks: [generate and save the model](https://nbviewer.jupyter.org/github/dongzhang84/Featuretools/blob/main/Titanic_Featuretools_automation_train.ipynb), and [load model and predict](https://github.com/dongzhang84/Featuretools/blob/main/Titanic_Featuretools_automation_test.ipynb). 

## Load FeatureTools

In the Titanic problem, I split the training dataset to X_train (**original features**) and y_train (**labels**):

```python
y_train = X_train.Survived
X_train = X_train.drop(['Survived'], axis=1)
```

The next step is to load **FeatureTools** and generate entities:

```python
es = ft.EntitySet(id = 'titanic_data')
es = es.entity_from_dataframe(entity_id = 'df', dataframe = X_train, 
                              variable_types = 
                              {
                                  'Embarked': ft.variable_types.Categorical,
                                  'Sex': ft.variable_types.Boolean,
                                  'Title': ft.variable_types.Categorical,
                                  'Family_Size': ft.variable_types.Numeric,
                              },
                              index = 'Id')
```

One should add (categorical) features which can later be applied **One Hot** encoding method, for example:

```python
es = es.normalize_entity(base_entity_id='df', new_entity_id='Pclass', index='Pclass')
es = es.normalize_entity(base_entity_id='df', new_entity_id='Sex', index='Sex')
es = es.normalize_entity(base_entity_id='df', new_entity_id='Embarked', index='Embarked')
es = es.normalize_entity(base_entity_id='df', new_entity_id='Title', index='Title')
es = es.normalize_entity(base_entity_id='df', new_entity_id='Deck', index='Deck')
```

Note that base_entity_id='df' should be the same. 



## Feature Enigneering

Then one can use **FetureTools** for automatic feature engineering: 

```python
feature_matrix, feature_names = ft.dfs(entityset=es, 
                                       target_entity = 'df',
                                       max_depth = 2, 
                                       ignore_variables={'df':['Survived','PassengerId']})
```

This is not enough. In order to save features, one needs to do

```python
feature_matrix_enc, features_enc = ft.encode_features(feature_matrix, feature_names, include_unknown=False)
```

Not sure if "include_unknown=False" is necessary. 

Save features by

```python
ft.save_features(features_enc, "titanic/feature_definitions.json")
```

and let

```python
X_train = feature_matrix_enc.copy()
```



To encode categorical data, one needs to find all categorical columns. Then do

```python
from sklearn.preprocessing import OrdinalEncoder

encode_list = []

for col in categorical_columns:
    if col in X_train.columns:
        le = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        le.fit(list(X_train[col].astype(str).values.reshape(-1, 1)))
        encode_list.append(le)
        X_train[col] = le.transform(list(X_train[col].astype(str).values.reshape(-1, 1)))
```

and save the encoders:

```python
with open("titanic/models.pkl", "wb") as f:
    pickle.dump(encode_list, f)
```

To write a new pickle file one needs to use "wb", to append to use "ab". 



## Feature Selection

There are multiple ways to do feature selection. For example, [Recursive Feature Elimination (RFE) with Random Forest](https://github.com/dongzhang84/Featuretools/blob/main/Titanic_automation_train_v2.ipynb), and [Recursive Feature Elimination (RFE) with Logistic Regression](https://github.com/dongzhang84/Featuretools/blob/main/Titanic_automation_train_v3.ipynb). 

Here I provide a templete to remove collinear features:

```python
# Threshold for removing correlated variables
threshold = 0.8

# Absolute value correlation matrix
corr_matrix = X_train.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Select columns with correlations above threshold
collinear_features = [column for column in upper.columns if any(upper[column] > threshold)]

print('There are %d features to remove.' % (len(collinear_features)))
```

```python
train_fe = X_train.drop(columns = collinear_features)
train_fe
```

Save the selected features: 

```python
FE_option = train_fe.columns

with open("titanic/models.pkl", "ab") as f:
    pickle.dump(FE_option1, f)
```

## Modeling

Working on **train_fe**, modeling is straightforward. One example:

```python
# Tuning Random Forest model for features "features_set", makes prediction and save it into file  
train_fe = X_train.drop(columns = collinear_features)

random_forest = GridSearchCV(estimator=RandomForestClassifier(), 
                             param_grid={'n_estimators': [100, 500]}, cv=5).fit(train_fe, y_train)
random_forest.fit(train_fe, y_train)
random_forest.score(train_fe, y_train)
acc_random_forest = round(random_forest.score(train_fe, y_train) * 100, 2)
print(acc_random_forest)
```

Do not forget to save the model:

```python
with open("titanic/models.pkl", "ab") as f:
    pickle.dump(random_forest, f)
```



# Load Model

For model deployment, one has

- Saved features from **FeatureTools** automatic feature engineering.
- A pickle file including categorical data feature engineering, selected features and the final model. 



**(1)** Load everything by:

```python
pickle_list = []

with open("titanic/models.pkl", "rb") as f:
    while True:
        try:
            pickle_list.append(pickle.load(f))
        except EOFError:
            break
```

and

```python
encode_list = pickle_list[0]
selected_features = pickle_list[1]
model = pickle_list[2]
```



**(2)** To reproduce the automazed feature engineering by **FeatureTools**:

```python
y_test = X_test.Survived
X_test = X_test.drop(['Survived'], axis=1)
```

Load **FeatureTools**:

```python
es_tst = ft.EntitySet(id = 'titanic_data')
es_tst = es_tst.entity_from_dataframe(entity_id = 'df', dataframe = X_test, 
                              variable_types = 
                              {
                                  'Embarked': ft.variable_types.Categorical,
                                  'Sex': ft.variable_types.Boolean,
                                  'Title': ft.variable_types.Categorical,
                                  'Family_Size': ft.variable_types.Numeric,
                              },
                              index = 'Id')

es_tst = es_tst.normalize_entity(base_entity_id='df', new_entity_id='Pclass', index='Pclass')
es_tst = es_tst.normalize_entity(base_entity_id='df', new_entity_id='Sex', index='Sex')
es_tst = es_tst.normalize_entity(base_entity_id='df', new_entity_id='Embarked', index='Embarked')
es_tst = es_tst.normalize_entity(base_entity_id='df', new_entity_id='Title', index='Title')
es_tst = es_tst.normalize_entity(base_entity_id='df', new_entity_id='Deck', index='Deck')
```

and load saved features:

```python
feature_matrix_tst = ft.calculate_feature_matrix(features=saved_features, entityset=es_tst)

X_test = feature_matrix_tst.copy()
```



**(3)** Revisit Categorical Feature engineering by:

```python
i = 0

for col in categorical_columns:
    
    try: 
        X_test[col] = encode_list[i].transform(list(X_test[col].astype(str).values.reshape(-1, 1)))
    except:
        print(col, "An exception occurred")
    i += 1
```

So each saved encoders work on the corresponding column. 



**(4) ** Selected features:

```python
test_fe = X_test[selected_features]
test_fe.fillna(0, inplace=True)
```

The second line above is not necessary, and can be optimized. 



**(5)** Finally use the loaded model to do prediction:

```python
Y_pred = model.predict(test_fe).astype(int)
```

