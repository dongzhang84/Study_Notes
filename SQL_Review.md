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



### Limit, offset

```sql
SELECT 
    column_list
FROM
    table1
ORDER BY column_list
LIMIT row_count OFFSET offset;
```

For example:

```sql
SELECT 
    employee_id, first_name, last_name
FROM
    employees
ORDER BY first_name
LIMIT 5 OFFSET 3;
```

![SQL LIMIT OFFSET example](https://www.sqltutorial.org/wp-content/uploads/2016/04/SQL-LIMIT-OFFSET-example.png)



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



### Min, Max, Count, Avg, Sum

```sql
SELECT MIN(column_name)
FROM table_name
WHERE condition;
-----------------------------------------------------

SELECT MAX(column_name)
FROM table_name
WHERE condition;
-----------------------------------------------------

SELECT COUNT(column_name)
FROM table_name
WHERE condition;
-----------------------------------------------------

SELECT AVG(column_name)
FROM table_name
WHERE condition;
-----------------------------------------------------

SELECT SUM(column_name)
FROM table_name
WHERE condition;
```



### Like

```sql
SELECT column1, column2, ...
FROM table_name
WHERE columnN LIKE pattern;
```

![sql_2.png](https://github.com/dongzhang84/Study_Notes/blob/main/figures/SQL/sql_2.png?raw=true)

```sql
SELECT * FROM Customers
WHERE CustomerName LIKE 'a%';
```



### In

```sql
SELECT column_name(s)
FROM table_name
WHERE column_name IN (value1, value2, ...);
```

or

```sql
SELECT column_name(s)
FROM table_name
WHERE column_name IN (SELECT STATEMENT);
```

For example:

```sql
SELECT * FROM Customers
WHERE Country IN (SELECT Country FROM Suppliers);
```



### Between 

```sql
SELECT column_name(s)
FROM table_name
WHERE column_name BETWEEN value1 AND value2;
```



### Join

![sql_3.png](https://github.com/dongzhang84/Study_Notes/blob/main/figures/SQL/sql_3.png?raw=true)

```sql
SELECT column_name(s)
FROM table1
INNER JOIN table2
ON table1.column_name = table2.column_name;
```

```sql
SELECT column_name(s)
FROM table1
LEFT JOIN table2
ON table1.column_name = table2.column_name;
```

```sql
SELECT column_name(s)
FROM table1
RIGHT JOIN table2
ON table1.column_name = table2.column_name;
```

```sql
SELECT column_name(s)
FROM table1
FULL OUTER JOIN table2
ON table1.column_name = table2.column_name
WHERE condition;
```

Self Join

```sql
SELECT column_name(s)
FROM table1 T1, table1 T2
WHERE condition;
```



### Union

```sql
SELECT column_name(s) FROM table1
UNION
SELECT column_name(s) FROM table2;
```

The `UNION` operator selects only distinct values by default. To allow duplicate values, use `UNION ALL`

```sql
SELECT column_name(s) FROM table1
UNION ALL
SELECT column_name(s) FROM table2;
```



### Group by

```sql
SELECT column_name(s)
FROM table_name
WHERE condition
GROUP BY column_name(s)
ORDER BY column_name(s);
```

For example:

```sql
SELECT COUNT(CustomerID), Country
FROM Customers
GROUP BY Country
ORDER BY COUNT(CustomerID) DESC;
```



### Having

The `HAVING` clause was added to SQL because the `WHERE` keyword cannot be used with aggregate functions.

```sql
SELECT column_name(s)
FROM table_name
WHERE condition
GROUP BY column_name(s)
HAVING condition
ORDER BY column_name(s);
```



### Case

```sql
CASE
    WHEN condition1 THEN result1
    WHEN condition2 THEN result2
    WHEN conditionN THEN resultN
    ELSE result
END;
```





## Advanced SQL

### Window Function

[Reference1](https://medium.com/geekculture/window-functions-in-sql-b630b0c11a57), [Reference2](https://www.analyticsvidhya.com/blog/2020/12/window-function-a-must-know-sql-concept/)



General formula:

```sql
WINDOW_FUNCTION() OVER (
[PARTITION BY <COLUMN>]
[ORDER BY <COLUMN>]
[ ROWS BETWEEN N PRECEDING AND CURRENT ROW] OR [ROWS BETWEEN CURRENT AND N FOLLOWING])
AS
<ALIAS>
```

Short version:

```sql
WINDOW_FUNCTION() OVER (PARTITION BY <COLUMN>)
AS
<ALIAS>
```



![img](https://miro.medium.com/max/2000/1*7ApZu3fLL3zOo_2Cddd4lQ.png)

- DENSE_RANK(): Computes the rank of a certain element in a given list of elements or column where this would be consecutive to the previous rank.

- RANK(): Computes the rank of a certain element in a given list of elements or column where the current rank need not be of a consecutive rank to the previous rank.

- ROW_NUMBER(): returns the sequential number of a row within a partition of a result set, starting at 1 for the first row in each partition.



For example: 

![img](https://miro.medium.com/max/1400/1*GpI10pfTPUS-e49EfjDuZQ.png)

The result is:

![img](https://miro.medium.com/max/2000/1*l4tqXewI2tuKZigVmpYmFw.png)



**Lead and Lag**:

![LEAD window functions](https://cdn.analyticsvidhya.com/wp-content/uploads/2020/11/lead-function-sql.png)

![LAG window functions](https://cdn.analyticsvidhya.com/wp-content/uploads/2020/11/lag-function-sql.png)





### Find Median

[reference](https://stackoverflow.com/questions/1342898/function-to-calculate-median-in-sql-server)

```sql
SELECT
(
 (SELECT MAX(Score) FROM
   (SELECT TOP 50 PERCENT Score FROM Posts ORDER BY Score) AS BottomHalf)
 +
 (SELECT MIN(Score) FROM
   (SELECT TOP 50 PERCENT Score FROM Posts ORDER BY Score DESC) AS TopHalf)
) / 2 AS Median
```

