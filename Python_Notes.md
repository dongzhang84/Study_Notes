# Notes on Python coding for Data Science

## Content

**Pandas Data Manipulation**

- Remove warnings
- Show all columns
- Add rows
- Aggregation Function
- Function using multiple columns
- Outliers remover
- Quantile rankings



**Random Number Generator**

- Uniform Distribution
- Gaussian Distribution



**Matplotlib & Seaborn**

- plot basic: multiple panels
- plot basic: contour
- seaborn barplot (basic)
- seaborn bar plot with hues
- seaborn box plot

---



## Pandas Data Manipulation

#### Remove Warnings

```python
import warnings
warnings.filterwarnings("ignore")
```



#### Show all columns

```python
pd.set_option('display.max_columns', None)
```



#### Add  rows to pandas

See [this link](https://pythonexamples.org/pandas-dataframe-add-append-row/)

```Python
data = {'name': ['Somu', 'Kiku', 'Amol', 'Lini'],
	'physics': [68, 74, 77, 78],
	'chemistry': [84, 56, 73, 69],
	'algebra': [78, 88, 82, 87]}

	
#create dataframe
df_marks = pd.DataFrame(data)
print('Original DataFrame\n------------------')
print(df_marks)

new_row = {'name':'Geo', 'physics':87, 'chemistry':92, 'algebra':97}
#append row to the dataframe
df_marks = df_marks.append(new_row, ignore_index=True)

print('\n\nNew row added to DataFrame\n--------------------------')
print(df_marks)
```



#### Aggregation Function

The simplest case for a single aggregation:

```
df.groupby('A')['B'].count()
df.groupby('A')['B'].sum()
df.groupby('A')['B'].mean()
```

For multiple aggregations:

```python
# Sample database

df = pd.DataFrame(
    {"A": [1, 1, 2, 2],
     "B": [1, 2, 3, 4],
     "C": [0.362838, 0.227877, 1.267767, -0.562860],
    }
)
```

Aggregation of one column:

```python
df.groupby('A').B.agg(['min', 'max', 'sum', 'mean', 'median', 'count']).reset_index()
```

|      |    A |  min |  max |  sum | mean | median | count |
| ---: | ---: | ---: | ---: | ---: | ---: | -----: | ----: |
|    0 |    1 |    1 |    2 |    3 |  1.5 |    1.5 |     2 |
|    1 |    2 |    3 |    4 |    7 |  3.5 |    3.5 |     2 |



Aggregation of multiple columns:

```python
df.groupby('A').agg({'B': ['min', 'max','count'], 'C': ['sum','median']})
```

|      |    B |      |       |        C |          |
| ---: | ---: | ---: | ----: | -------: | -------: |
|      |  min |  max | count |      sum |   median |
|    A |      |      |       |          |          |
|    1 |    1 |    2 |     2 | 0.590715 | 0.295357 |
|    2 |    3 |    4 |     2 | 0.704907 | 0.352454 |



#### Function Using Multiple Columns

Define a function  f(x1, x2, x3)

```python
df['col_new'] = df.apply(lambda x: f(x.col_1, x.col_2, x.col_3), axis=1)
```



#### Outliers Remover

````python
Q1 = df[field].quantile(0.25)
Q3 = df[field].quantile(0.75)
IQR = Q3 - Q1
minimal = Q1 - 1.5*IQR
maximal = Q3 + 1.5*IQR

print(Q1,Q3,minimal,maximal)

outliers = df[(df[field] <= minimal) & (df[field] >= maximal)]
````



#### Quantile Ranking

This is one example to make quantile ranking based on a random list:

```Python
from scipy import stats

df = pd.DataFrame()
x = np.random.uniform(0,1,1000)
ranking = [stats.percentileofscore(x, a, 'rank')/100 for a in x]

df['x'] = x
df['ranking'] = ranking

df
```





----

## Random Number Generator

#### Uniform Distribution

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

#### Gaussian Distribution

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

2D Gaussian Distribution

```Python
mean = [0.5, 0.5]
cov = [[0.2*0.2, 0.], [0., 0.1*0.1]] 
random_list = np.random.multivariate_normal(mean, cov, N)
```





## Matplotlib & Seaborn



Import the libraries:

```python
import matplotlib.pyplot as plt
import seaborn as sns
```



Plot multiple panels:

```python
# generate a random data set 

N = 1000

random_list1 = np.random.uniform(0,1,size=(N,2))
random_list2 = np.concatenate((random_list1, np.random.uniform(0,1,size=(N,2))))
random_list3 = np.concatenate((random_list2, np.random.uniform(0,1,size=(3 * N,2))))
random_list4 = np.concatenate((random_list3, np.random.uniform(0,1,size=(5 * N,2))))

random_list = [random_list1, random_list2, random_list3, random_list4]

# plot multiple panels

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(9, 8.5))

i = 0
for row in ax:
  for col in row:
     
    col.scatter(random_list[i][:,0], random_list[i][:,1], c='b', s=3)
    col.set_xlim(0,1)
    col.set_ylim(0,1)

    i += 1

plt.show()
```



![multiple_panels.png](https://github.com/dongzhang84/Study_Notes/blob/main/figures/python_notes/multiple_panels.png?raw=true)

------

#### contour

```python
df= pd.DataFrame(data={'M': [10,20,30,40,50],
                       '10':[-0.029, -0.13, -0.58, -1.55, -2.82],
                       '20':[-0.014, -0.062, -0.29, -1.01, -2.16],
                       '30':[-0.0059, -0.0237, -0.106, -0.464, -1.37],
                       '50':[-0.0031, -0.0123, -0.051, -0.223, -0.854]})
```

The table is:

|      |    M |     10 |     20 |      30 |      50 |
| ---: | ---: | -----: | -----: | ------: | ------: |
|    0 |   10 | -0.029 | -0.014 | -0.0059 | -0.0031 |
|    1 |   20 | -0.130 | -0.062 | -0.0237 | -0.0123 |
|    2 |   30 | -0.580 | -0.290 | -0.1060 | -0.0510 |
|    3 |   40 | -1.550 | -1.010 | -0.4640 | -0.2230 |
|    4 |   50 | -2.820 | -2.160 | -1.3700 | -0.8540 |



Make the contour plot:

```
X=[10, 20, 30, 50]
Y = df.M.values
Z = df.iloc[:,1:].values

fig,ax = plt.subplots(figsize=(7.5, 6))
contourf_ = plt.contourf(X,Y,Z, 100)
cbar = fig.colorbar(contourf_)
plt.show()
```



![plot_contour_1.png](https://github.com/dongzhang84/Study_Notes/blob/main/figures/python_notes/plot_contour_1.png?raw=true)

-------

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

----

#### seaborn barplot

Load the iris data:

```Python
import seaborn as sns
import matplotlib.pyplot as plt

df = sns.load_dataset('iris')

sns.boxplot(y=df["sepal_length"])
plt.show()
```

![seaborn_box_1.png](https://github.com/dongzhang84/Study_Notes/blob/main/figures/python_notes/seaborn_box_1.png?raw=true)

Another plot: 

```python
sns.boxplot( y=df["species"], x=df["sepal_length"] );
plt.show()
```

![seaborn_box_2.png](https://github.com/dongzhang84/Study_Notes/blob/main/figures/python_notes/seaborn_box_2.png?raw=true)

See [this link](https://pythonbasics.org/seaborn-boxplot/) for more details. 