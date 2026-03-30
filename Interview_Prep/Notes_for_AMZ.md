

### Convert Timestamp to Date Format

```python
from datetime import datetime, timedelta

df['orderDate']=df.orderDate.apply(lambda x: datetime.fromtimestamp(x))
```

### Calculate Time Interval (in seconds)

```python

 df['time_interval'] = (df['time1'] - df['time2']).dt.total_seconds()

```

### Convert Datetime to Strings
```
df_us['year'] = df_us['order_day'].dt.year
df_us['week'] = df_us['order_day'].dt.isocalendar().week
df_us['weekofyear'] = df_us['year'].astype('str') + df_us['week'].astype('str').str.zfill(2)
```

### Check Categorial Variables
```python
object_list = list(df_train.select_dtypes(include=['object']).columns)
object_list
```

### Check Numerical Variables
```python
number_list = list(df.select_dtypes(include=['float64','int64']).columns)
len(number_list)
```

### Group and rename column in one line
```python
tmp = df.groupby(['weekofyear','week'])['C'].sum().reset_index(name='D')
```


### Query Time Format

```SQL
query = """
select *
from schema.table
where date_time >= '%s'
and date_time < '%s'
""" % (start_date, end_date)
```

## AutoGluon Template
```
!pip install autogluon
```

```
from autogluon.tabular import TabularDataset, TabularPredictor
```
```
df_train = pd.concat([df_train,df_val],axis=0)
```

```
eval_metric = 'roc_auc'
label = '__tag__'

predictor = TabularPredictor(label=label, eval_metric=eval_metric)\
.fit(df_train,hyperparameters={'GBM':{'ag_args_fit': {'max_memory_usage_ratio': 1.2}}, 
                               'XGB':{'ag_args_fit': {'max_memory_usage_ratio': 1.2}}, 
                               'XT':{'ag_args_fit': {'max_memory_usage_ratio': 1.2}}})
```

```
%%time
predictor.leaderboard(df_test, silent=True)
```


## Command and AWS

compress a directory

```
tar -czvf abcd.tar.gz .
```
or
```
tar -cvzf abcd.tar.gz *.csv
```

unzip file

```
tar -zxvf abcd.tar.gz
```

sync directory

```
!aws s3 sync directory1 directory2
```

remove directory
```
!aws s3 rm s3://bucket/folder --recursive
```
