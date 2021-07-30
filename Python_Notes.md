# Notes on Python coding for Data Science

## Content

**Random Number Generator**

- Uniform Distribution
- Gaussian Distribution

**Matplotlib & Seaborn**

- seaborn barplot (basic)
- seaborn bar plot with hues





------

### Random Number Generator

**Uniform Distribution**:

```
from random import *

x_list = []
y_list = []

for n in range(5000):
  x = uniform(0,1)
  y = uniform(0,1)
  x_list.append(x)
  y_list.append(y)
```



Linear regression and Plot

```Python
# Linear Regression Fit
coef = np.polyfit(x_list,y_list,1)
poly1d_fn = np.poly1d(coef)

# Plot
plt.figure(figsize=(6, 6))          
plt.scatter(x_list,y_list, c='y', s=6)
plt.plot(x_list, poly1d_fn(x_list), '--k')
plt.xlim(0,1)
plt.ylim(0,1)
plt.show()
```

![random_1.png](https://github.com/dongzhang84/Study_Notes/blob/main/figures/python_notes/random_1.png?raw=true)

Note that we can use random.randint(a,b) to generate random integers between a and b. 


---------

**Gaussian Distribution**


$$
p(x) = \frac{1}{\sqrt{2\pi \sigma^2}}e^{-\frac{(x-\mu)^2}{2 \sigma^2}}
$$


Generate random numbers: 

```python
import numpy as np

# generte 1000 random number with Gaussian distribution

mu, sigma = 0, 0.1 
s = np.random.normal(mu, sigma, 1000)
count, bins, ignored = plt.hist(s, 30, density=True)
```

![gaussian_1.png](https://github.com/dongzhang84/Study_Notes/blob/main/figures/python_notes/gaussian_1.png?raw=true)

Visualization: 

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 6))
count, bins, ignored = plt.hist(s, 30, density=True)
plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
               np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
         linewidth=2, color='r')
plt.show()
```

![gaussian_2.png](https://github.com/dongzhang84/Study_Notes/blob/main/figures/python_notes/gaussian_2.png?raw=true)





## Matplotlib & Seaborn



Import the libraries:

```python
import matplotlib.pyplot as plt
import seaborn as sns
```



#### seaborn barplot (basic)

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

Add a line: 

```
plt.axhline(y=6000, color='r', linestyle='--')
```

The plot becomes:

![seaborn_bar_2.png](https://github.com/dongzhang84/Study_Notes/blob/main/figures/python_notes/seaborn_bar_2.png?raw=true)

----

#### seaborn bar plot with hues

Create another pandas data frame: 

```
df = pd.DataFrame(data = {'Size': ['Small', 'Mid', 'Large', 'Xlarge'],
                          'Count A': [162110, 125000, 69000, 55000],
                          'Rate A': [0.44, 0.25, 0.22, 0.21],
                          'Count B': [81000, 56000, 20000, 4800],
                          'Rate B': [0.22, 0.14, 0.27, 0.32]})
```

|      |   Size | Count  A | Rate A | Count B | Rate B |
| ---: | -----: | -------: | -----: | ------: | ------ |
|    0 |  Small |   162110 |   0.44 |   81000 | 0.22   |
|    1 |    Mid |   125000 |   0.25 |   56000 | 0.14   |
|    2 |  Large |    69000 |   0.22 |   20000 | 0.27   |
|    3 | Xlarge |    55000 |   0.21 |    4800 | 0.32   |

In order to plot bars with hue (e.g, A vs B in this case), need to convert the df to the following:

```
df_convert = pd.melt(df[['Size','Rate A','Rate B']], 
                  id_vars="Size", var_name="A/B", value_name="Rate")
```

|      |   Size |   A/ B | Rate |
| ---: | -----: | -----: | ---- |
|    0 |  Small | Rate A | 0.44 |
|    1 |    Mid | Rate A | 0.25 |
|    2 |  Large | Rate A | 0.22 |
|    3 | Xlarge | Rate A | 0.21 |
|    4 |  Small | Rate B | 0.22 |
|    5 |    Mid | Rate B | 0.14 |
|    6 |  Large | Rate B | 0.27 |
|    7 | Xlarge | Rate B | 0.32 |

Then make the seaborn plot with hue:

```
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x='Size', y='Rate', hue='A/B', data=df_convert, ax=ax)
plt.xlabel('Size', fontsize=15)
plt.ylabel('Rate',fontsize=15)
plt.xticks(rotation=0, fontsize='x-large')
plt.yticks(rotation=0, fontsize='x-large')
plt.legend(prop={'size': 20})
plt.show()
```

The plot is:

![seaborn_bar_hue_1.png](https://github.com/dongzhang84/Study_Notes/blob/main/figures/python_notes/seaborn_bar_hue_1.png?raw=true)



Another way to plot a more complex plot combined multiple fields (A/B count and rate) is as follows ([reference](https://stackoverflow.com/questions/32474434/trying-to-plot-a-line-plot-on-a-bar-plot-using-matplotlib)):

```
plt.figure(figsize=(10, 6))          
N = 4
count_A = list(df['Count A'])
width = 0.4       # the width of the bars
count_B = list(df['Count B'])

ind = np.arange(N)
plt.bar(ind, count_A, width, color='b', label='Count A')
plt.bar(ind+width, count_B, width, color='orange', label='Count B')
plt.ylabel('Count A/B') 
plt.xticks([0,1,2,3])
plt.xlabel('Size', fontsize=15)
plt.ylabel('Rate',fontsize=15)
plt.xticks(rotation=0, fontsize='x-large')
plt.yticks(rotation=0, fontsize='x-large')

x = np.arange(N)
y1 = list(df['Rate A'])
y2 = list(df['Rate B'])

ax2 = plt.twinx()
ax2.plot(x, y1, color='b', label='Rate A')
ax2.plot(x, y2, color='orange', label='Rate B')
ax2.set_ylim(0, 0.5)
ax2.set_ylabel('Rate', fontsize=15)
ax2.legend(prop={'size': 15})


plt.show()
```

The above plot code is very useful, and the result is:

![seaborn_bar_hue_2.png](https://github.com/dongzhang84/Study_Notes/blob/main/figures/python_notes/seaborn_bar_hue_2.png?raw=true)