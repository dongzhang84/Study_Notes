

### Convert Timestamp to Date Format

```python
from datetime import datetime, timedelta

df['orderDate']=df.orderDate.apply(lambda x: datetime.fromtimestamp(x))
```

### Calculate Time Interval (in seconds)

```python

 df['time_interval'] = (df['time1'] - df['time2']).dt.total_seconds()

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
tar -czvf abcd.tar.gz abcd/
```

unzip file

```
tar -zxvf abcd.tar.gz
```
sync directory

```
!aws s3 sync directory1 directory2
```
