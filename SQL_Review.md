# SQL Review



## Basic SQL

### Select

```sql
SELECT *
FROM table_name;
```

```sql
SELECT column1, column2, ...
FROM table_name;
```

```sql
SELECT DISTINCT column1, column2, ...
FROM table_name;
```



### Where

```sql
SELECT column1, column2, ...
FROM table_name
WHERE condition;
```

Check Operators in The WHERE Clause

![sql_1.png](https://github.com/dongzhang84/Study_Notes/blob/main/figures/SQL/sql_1.png?raw=true)

```python
SELECT column1, column2, ...
FROM table_name
WHERE condition1 AND condition2 AND condition3 ...;

-----------------------------------------------------

SELECT column1, column2, ...
FROM table_name
WHERE condition1 OR condition2 OR condition3 ...;

-----------------------------------------------------

SELECT column1, column2, ...
FROM table_name
WHERE NOT condition;
```



### Order By

```sql
SELECT column1, column2, ...
FROM table_name
ORDER BY column1, column2, ... ASC|DESC;
```



### Null Values

```sql
SELECT column_names
FROM table_name
WHERE column_name IS NULL;
```

```sql
SELECT column_names
FROM table_name
WHERE column_name IS NOT NULL;
```



### Update

```sql
UPDATE table_name
SET column1 = value1, column2 = value2, ...
WHERE condition;
```

For example: 

```sql
UPDATE Customers
SET ContactName = 'Alfred Schmidt', City= 'Frankfurt'
WHERE CustomerID = 1;
```



### Delete

Note that it start with **delete from**:

```sql
DELETE FROM table_name 
WHERE condition;
```



### Min & Max

```sql
SELECT MIN(column_name)
FROM table_name
WHERE condition;
-----------------------------------------------------

SELECT MAX(column_name)
FROM table_name
WHERE condition;
```

