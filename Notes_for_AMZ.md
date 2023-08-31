

### Convert Timestamp to Date Format

```python
from datetime import datetime, timedelta

df['orderDate']=df.orderDate.apply(lambda x: datetime.fromtimestamp(x))
```

### Calculate Time Interval (in seconds)

```python

 df['time_interval'] = (df['time1'] - df['time2']).dt.total_seconds()

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

## Command and AWS

compress a directory

```
tar -czvf abcd.tar.gz .
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
