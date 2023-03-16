

### Convert Timestamp to Date Format

```python
from datetime import datetime, timedelta

df['orderDate']=df.orderDate.apply(lambda x: datetime.fromtimestamp(x))
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
