# Notes on Python coding for Data Science





## Matplotlib & Seaborn



Import the libraries:

```python
import matplotlib.pyplot as plt
import seaborn as sns
```



#### seaborn barplot

Create a pandas dataframe

```python
df = pd.DataFrame(data = {'ID': ['A','B','C','D','E','F','G','H','I','J'], 
                          'Rate': [5764,3809,2233,6239,2806,6269,4860,8822,3658,3193]})
```

Plot the distribution of "Rate" based on "ID":

```Python
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x = df['ID'], y = df['Rate'])
plt.xlabel('ID', fontsize=15)
plt.ylabel('Rate',fontsize=15)
plt.xticks(rotation=0, fontsize='x-large')
plt.yticks(fontsize=15)

plt.show()
```

The plot is like this:

![seaborn_bar_1.png](https://github.com/dongzhang84/Study_Notes/blob/main/figures/python_notes/seaborn_bar_1.png?raw=true)

