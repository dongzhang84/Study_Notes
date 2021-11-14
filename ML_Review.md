

# Machine Learning Review

## Traditional Machine Learning

- [Logistic regression](#Logistic Regression)



<a id='Logistic Regression'></a>

### Logistic Regression

See this [reference](https://towardsdatascience.com/optimization-loss-function-under-the-hood-part-ii-d20a239cde11): 

An example of Logistic model:

![{\displaystyle \ell =\log _{b}{\frac {p}{1-p}}=\beta _{0}+\beta _{1}x_{1}+\beta _{2}x_{2}}](https://wikimedia.org/api/rest_v1/media/math/render/svg/4a5e86f014eb1f0744e280eb0d68485cb8c0a6c3)

For more general consideration:

![img](https://miro.medium.com/max/2000/1*WP6xNUvfdtHgjcjnOnQETw.png)

![img](https://miro.medium.com/max/1712/1*P1Wu65ic5sK8Jhq8Sl-WxQ.png)

![img](https://miro.medium.com/max/2404/1*LFUX3uWdiZ-UTc5Vl4RVjA.png)





**Cost Function**

![img](https://miro.medium.com/max/2000/1*PT9WfxoXFuE-2yiYrEu9Zg.png)

Or written as:

![img](https://miro.medium.com/max/1400/1*dEZxrHeNGlhfNt-JyRLpig.png)

**L1 and L2 Regularization**

![img](https://miro.medium.com/max/1400/1*vwhvjVQiEgLcssUPX6vxig.png)

* L1: Lasso

* L2: Ridge



**Gradient Descent**

![img](https://i.stack.imgur.com/zgdnk.png)



### Gradient Descent and other Methods

See this [reference](https://medium.com/@lachlanmiller_52885/machine-learning-week-1-cost-function-gradient-descent-and-univariate-linear-regression-8f5fe69815fd):

![img](https://miro.medium.com/max/1400/1*WGHn1L4NveQ85nn3o7Dd2g.png)

![img](https://miro.medium.com/max/1020/1*QKHtyn4Rr-0R-s0an1eSsA.png)



#### Stochastic Gradient Descent

Mini-batch gradient descent 

Mock code [reference1](https://www.zhihu.com/question/264189719):

[reference2](https://ruder.io/optimizing-gradient-descent/):

**Gradient Descent**

```python
repeat until convergence:
  
  theta_j -= theta_j - alpha * dL_dtheta_j
```

![img](https://pic1.zhimg.com/80/v2-5809743fd06c4ff804753d29e4b83935_1440w.jpg?source=1940ef5c)



```python
for i in range(nb_epochs):
  params_grad = evaluate_gradient(loss_function, data, params)
  params = params - learning_rate * params_grad
```





**Stochastic Gradient Descent**

```python
repeat until convergence or for i in range(epoch):
  
  theta_j -= theta_j - alpha * dL_dtheta_j(i in sample_i)
```

![img](https://pica.zhimg.com/80/v2-b3f14a09ad27df9c66a3af208060f5d7_1440w.jpg?source=1940ef5c)

```python
for i in range(nb_epochs):
  np.random.shuffle(data)
  for example in data:
    params_grad = evaluate_gradient(loss_function, example, params)
    params = params - learning_rate * params_grad
```







### Overfitting vs Underfitting

- **Overfitting**: Good performance on the training data, poor generliazation to other data.
- **Underfitting**: Poor performance on the training data and poor generalization to other data

![img](https://docs.aws.amazon.com/machine-learning/latest/dg/images/mlconcepts_image5.png)

[How to deal with overfitting](https://towardsdatascience.com/8-simple-techniques-to-prevent-overfitting-4d443da2ef7d):

1. **Hold-out**

2. **Cross-validation**

3. **Data augmentation**

   ![img](https://miro.medium.com/max/1400/0*JgP_DG16kisBAdpS.png)

4. **Feature selection**
   ![img](https://miro.medium.com/max/1400/0*N3paES6IzJ8oyh9p)

5. **L1 / L2 regularization**
   ![img](https://miro.medium.com/max/1400/0*69Jgv2gwAPtOIwNh.png)

6. **Remove layers / number of units per layer**

7. **Dropout**
   ![img](https://miro.medium.com/max/1400/0*YCofAkhSErYvlpRT.png)

8. **Early stopping**

   ![img](https://miro.medium.com/max/920/0*b4lf4K0PswVYZXdI.png)

#### Bias vs Variance Trade Off

[reference](https://towardsdatascience.com/understanding-the-bias-variance-tradeoff-165e6942b229)

**What is bias?**

Bias is the difference between the average prediction of our model and the correct value which we are trying to predict. 

**What is variance?**

Variance is the variability of model prediction for a given data point or a value which tells us spread of our data. Model with high variance pays a lot of attention to training data and does not generalize on the data which it hasn’t seen before.



![img](https://miro.medium.com/max/580/1*BtpFTBrGaQNE3TvU-0EVSQ.png)

![img](https://miro.medium.com/max/1158/1*e7VaoBh5apjaM2p4afkFyg.png)

![img](https://miro.medium.com/max/936/1*xwtSpR_zg7j7zusa4IDHNQ.png)

![img](https://miro.medium.com/max/1124/1*RQ6ICt_FBSx6mkAsGVwx8g.png)







### Classification Metrics

[reference](https://towardsdatascience.com/the-5-classification-evaluation-metrics-you-must-know-aa97784ff226)

Binary Classification:

![confusion_matrix.png](https://github.com/dongzhang84/Study_Notes/blob/main/figures/ML/confusion_matrix.png?raw=true)

#### Accuracy, Precision, and Recall

```
Accuracy = (TP+TN)/(TP+FP+FN+TN)

Precision = (TP)/(TP+FP)

Recall = (TP)/(TP+FN)
```

![img](https://miro.medium.com/max/752/0*-lZUM_HsT3RsgePy.png)

Weighted F1 score:

![img](https://miro.medium.com/max/744/0*Z4hC6C2XOV192LgE)

#### AUC

```
Sensitivty = TPR(True Positive Rate)= Recall = TP/(TP+FN)

1- Specificity = FPR(False Positive Rate)= FP/(TN+FP)
```



![img](https://miro.medium.com/max/1400/0*Dkq9YLJM5t8k0Fcb)



### Cross Validation

![img](https://upload.wikimedia.org/wikipedia/commons/thumb/b/b5/K-fold_cross_validation_EN.svg/2880px-K-fold_cross_validation_EN.svg.png)



### Decision Tree

A decision tree is a map of the possible outcomes of a series of related choices.

![decision tree](https://d2slcw3kip6qmk.cloudfront.net/marketing/pages/chart/seo/decision-tree/discovery/sample-decision-tree.svg)

#### Metrics

**Gini Impurity**

![{\displaystyle \operatorname {I} _{G}(p)=\sum _{i=1}^{J}\left(p_{i}\sum _{k\neq i}p_{k}\right)=\sum _{i=1}^{J}p_{i}(1-p_{i})=\sum _{i=1}^{J}(p_{i}-p_{i}^{2})=\sum _{i=1}^{J}p_{i}-\sum _{i=1}^{J}p_{i}^{2}=1-\sum _{i=1}^{J}p_{i}^{2}}](https://wikimedia.org/api/rest_v1/media/math/render/svg/fe3533c9142312ae339b6e1b9c1d8bffd3bd33b1)

**Information Gain**

![{\displaystyle \mathrm {H} (T)=\operatorname {I} _{E}\left(p_{1},p_{2},\ldots ,p_{J}\right)=-\sum _{i=1}^{J}p_{i}\log _{2}p_{i}}](https://wikimedia.org/api/rest_v1/media/math/render/svg/52f792af48b1a164791d2c5eeb2ba10d460d82d6)





### Random Forest

[reference](https://towardsdatascience.com/understanding-random-forest-58381e0602d2)

![img](https://miro.medium.com/max/960/0*Od-Z7H_-t_ipQDV6)



- **Definition**: Random forest, like its name implies, consists of a large number of individual decision trees that operate as an ensemble. Each individual tree in the random forest spits out a class prediction and the class with the most votes becomes our model’s prediction (see figure below).

- **The principle**: **The wisdom of crowds**. A large number of relatively uncorrelated models (trees) operating as a committee will outperform any of the individual constituent models.
- **Feature Randomness**: we end up with trees that are not only trained on different sets of data (thanks to **bagging**) but also use different features to make decisions.

[Demo](https://medium.com/@harshdeepsingh_35448/understanding-random-forests-aa0ccecdbbbb):



![img](https://miro.medium.com/max/1400/1*l16JAxJR5MJea12jut-FLQ.png)

![img](https://miro.medium.com/max/1400/1*5vlUF8FRR6flPPWK4wt-Kw.png)





#### RandomForestClassifier Parameters

- **n_estimators**,default=100. 
  The number of trees in the forest.

- **criterion**, **{“gini”, “entropy”}, default=”gini”**: 
  The function to measure the quality of a split. Supported criteria are “gini” for the Gini impurity and “entropy” for the information gain. Note: this parameter is tree-specific.

- **max_depth**, **default=None**: 
  The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.

- **bootstrap**, **bool, default=True**: 
  Whether bootstrap samples are used when building trees. If False, the whole dataset is used to build each tree.

- **class_weight**, **{“balanced”, “balanced_subsample”}, dict or list of dicts, default=None**:

  Weights associated with classes in the form `{class_label: weight}`. If not given, all classes are supposed to have weight one. For multi-output problems, a list of dicts can be provided in the same order as the columns of y.

- **max_features**, **{“auto”, “sqrt”, “log2”}, int or float, default=”auto”**:The number of features to consider when looking for the best split:

  - If int, then consider `max_features` features at each split.
  - If float, then `max_features` is a fraction and `round(max_features * n_features)` features are considered at each split.
  - If “auto”, then `max_features=sqrt(n_features)`.
  - If “sqrt”, then `max_features=sqrt(n_features)` (same as “auto”).
  - If “log2”, then `max_features=log2(n_features)`.
  - If None, then `max_features=n_features`.



#### Advantages of Random Forest

**1.** Random Forest is based on the **bagging** algorithm and uses **Ensemble Learning** technique. It creates as many trees on the subset of the data and combines the output of all the trees. In this way it **reduces overfitting** problem in decision trees and also **reduces the variance** and therefore **improves the accuracy**.

**2.** Random Forest can be used to **solve both classification as well as regression problems**.

**3.** Random Forest works well with both **categorical and continuous variables**.

**4.** Random Forest can automatically **handle missing values**.

**5. No feature scaling required:** No feature scaling (standardization and normalization) required in case of Random Forest as it uses rule based approach instead of distance calculation.

**6. Handles non-linear parameters efficiently:** Non linear parameters don't affect the performance of a Random Forest unlike curve based algorithms. So, if there is high non-linearity between the independent variables, Random Forest may outperform as compared to other curve based algorithms.

**7.** Random Forest can automatically **handle missing values**.

**8.** Random Forest is usually **robust to outliers** and can handle them automatically.

**9.** Random Forest algorithm is very **stable**. Even if a new data point is introduced in the dataset, the overall algorithm is not affected much since the new data may impact one tree, but it is very hard for it to impact all the trees.

**10.** Random Forest is comparatively **less impacted by noise**.





### Bootstrapping, Bagging and Boosting

[reference](https://towardsdatascience.com/ensemble-methods-bagging-boosting-and-stacking-c9214a10a205):

#### Bootstrapping

Bootstrapping is any test or metric that uses random sampling with replacement (e.g. mimicking the sampling process), and falls under the broader class of resampling methods. Bootstrapping assigns measures of accuracy (bias, variance, confidence intervals, prediction error, etc.) to sample estimates. This technique allows estimation of the sampling distribution of almost any statistic using random sampling methods.

![img](https://miro.medium.com/max/1400/1*lWnm3eJVe3uo95OcSg5jUA@2x.png)



![img](https://miro.medium.com/max/2000/1*7XVde-bMixpKf8mj61qhJQ@2x.png)

#### Bagging: Bootstrapping Aggregation

we can **fit a weak learner for each of these samples and finally aggregate them such that we kind of “average” their outputs**.

Assuming that we have L bootstrap samples (approximations of L independent datasets) of size B denoted

![img](https://miro.medium.com/max/2000/1*nu96mPOtrXosJYgWA4Rvbw@2x.png)

we can fit L almost independent weak learners (one on each dataset)

![img](https://miro.medium.com/max/708/1*Dn6v09t5_L5cvADxJHJzHQ@2x.png)

and then aggregate them into some kind of averaging process in order to get an ensemble model with a lower variance. For example, we can define our strong model such that

![img](https://miro.medium.com/max/2000/1*jEbEHwvfUzAUI00muEAVGw@2x.png)



**Bagging Demo**:

![img](https://miro.medium.com/max/2000/1*zAMhmZ78a6V9W878zfk5eA@2x.png)

**Bagging Example: Random Forest**: Demo:

![img](https://miro.medium.com/max/2000/1*_B5HX2whbTs3DS8M6YBD_w@2x.png)



#### Boosting

Being **mainly focused at reducing bias**

![img](https://miro.medium.com/max/2000/1*VGSoqefx3Rz5Pws6qpLwOQ@2x.png)

Depending on the last step:

![img](https://miro.medium.com/max/892/1*YUJJ5nDbhBi0SkFeccsTxQ@2x.png)

![img](https://miro.medium.com/max/2000/1*6JbndZ2zY2c4QqS73HQ47g@2x.png)

#### Bagging vs Boosting

- Bagging is usually applied where the classifier is unstable and has a high variance. 
- Boosting is usually applied where the classifier is stable and simple and has high bias.









### Support Vector Machine

The objective of the support vector machine algorithm is to find a hyperplane in an N-dimensional space(N — the number of features) that distinctly classifies the data points.

Loss function of SVM:

![img](https://miro.medium.com/max/1400/1*GQAd28bK8LKOL2kOOFY-tg.png)

#### Maximal Margin Classifier

- Choosing the hyperplane that is farthest from the training observations. 
- This margin can be achieve using support vectors



#### SVM with a Hard Margin:

![img](https://www.baeldung.com/wp-content/uploads/sites/4/2021/07/fig1-300x232.png)

#### SVM with a Soft Margin:

![img](https://www.baeldung.com/wp-content/uploads/sites/4/2021/07/fig2-300x234.png)



#### Kernel

![kernel_svm.png](https://github.com/dongzhang84/Study_Notes/blob/main/figures/ML/kernel_svm.png?raw=true)



### KNN Code

[reference](https://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/)

```python
# Example of making predictions
from math import sqrt
 
# calculate the Euclidean distance between two vectors
def euclidean_distance(row1, row2):
	distance = 0.0
	for i in range(len(row1)-1):
		distance += (row1[i] - row2[i])**2
	return sqrt(distance)
 
# Locate the most similar neighbors
def get_neighbors(train, test_row, num_neighbors):
	distances = list()
	for train_row in train:
		dist = euclidean_distance(test_row, train_row)
		distances.append((train_row, dist))
	distances.sort(key=lambda tup: tup[1])
	neighbors = list()
	for i in range(num_neighbors):
		neighbors.append(distances[i][0])
	return neighbors
 
# Make a classification prediction with neighbors
def predict_classification(train, test_row, num_neighbors):
	neighbors = get_neighbors(train, test_row, num_neighbors)
	output_values = [row[-1] for row in neighbors]
	prediction = max(set(output_values), key=output_values.count)
	return prediction
```







### Unsupervised Learning

#### Metrics for Clustering

[reference1](https://towardsdatascience.com/performance-metrics-in-machine-learning-part-3-clustering-d69550662dc6), [reference2](https://towardsdatascience.com/how-to-evaluate-unsupervised-learning-models-3aa85bd98aa2). 

**1. Silhouette Score**

The Silhouette Score and Silhouette Plot are used to measure the separation distance between clusters. 

**2. Rand Index**

Another commonly used metric is the Rand Index. It computes a similarity measure between two clusters by considering all pairs of samples and counting pairs that are assigned in the same or different clusters in the predicted and true clusterings.

The formula of the Rand Index is:

![img](https://miro.medium.com/max/1378/0*6i-kV9SM-IHKgrxm)



**3. Adjusted Rand Index**

The raw RI score is then “adjusted for chance” into the ARI score using the following scheme:

![img](https://miro.medium.com/max/1346/0*o_4xLVQEl5H--ZAR)

**4. Calinski-Harabasz Index**





#### K-Mean Clustering



![K_mean.png](https://github.com/dongzhang84/Study_Notes/blob/main/figures/ML/K_mean.png?raw=true)



Algorithm [reference](https://realpython.com/k-means-clustering-python/)

![k means algorithm](https://files.realpython.com/media/kmeans-algorithm.a94498a7ecd2.png)







## Deep Learning

### Deep Learning Conceptions/Parameters 

See [this reference](https://medium.com/mlearning-ai/difference-between-the-batch-size-and-epoch-in-neural-network-2d2cb2a16734)



**Batch Size**

Batch size is **the number of samples that usually pass through the neural network at one time**. The batch size commonly referred to as mini-batch.



**epoch**

An epoch is a term used in machine learning that refers to the number of passes the machine learning algorithm has made over the entire training dataset.



**What is the difference between batch and epoch?**

- **Batch size**: The batch size is the number of samples processed before updating the model. The number of epochs represents the total number of passes through the training dataset.
- **Epoch**: It indicates the number of passes of the entire training dataset the machine learning algorithm has completed.



**Dropout**

The term “dropout” refers to dropping out units (both hidden and visible) in a neural network.

![img](https://miro.medium.com/max/1400/0*YCofAkhSErYvlpRT.png)



### Back Propagation

[reference](https://neptune.ai/blog/backpropagation-algorithm-in-neural-networks-guide):

The forward and backward phases are repeated from some epochs. In each epoch, the following occurs:

1. The inputs are propagated from the input to the output layer.
2. The network error is calculated.
3. The error is propagated from the output layer to the input layer.

![Backpropagation passes architecture](https://i1.wp.com/neptune.ai/wp-content/uploads/Backpropagation-passes-architecture.png?resize=434%2C414&ssl=1)





## Recommendation System

### Matrix Factorization

Matrix factorization is a simple embedding model. Given the feedback matrix A ∈Rm×n, where m is the number of users (or queries) and n is the number of items, the model learns:

- A user embedding matrix U∈Rm×d, where row i is the embedding for user i.
- An item embedding matrix V∈Rn×d, where row j is the embedding for item j.

![Illustration of matrix factorization using the recurring movie example.](https://developers.google.com/machine-learning/recommendation/images/Matrixfactor.svg)



[reference](https://medium.datadriveninvestor.com/how-to-built-a-recommender-system-rs-616c988d64b2)

![img](https://miro.medium.com/max/1400/1*bhsECuKCTSHOGapubG33JA.png)

Optimization: 

![img](https://miro.medium.com/max/594/1*vZ5PAfak7r7tT26acWSU4A.png)





### Singular Value Decomposition

Enter Singular Value Decomposition (SVD). SVD is a fancy way to factorizing a matrix into three other matrices (*A = UΣVᵀ*). The way SVD is done guarantees those 3 matrices carry some nice mathematical properties.



![img](https://miro.medium.com/max/736/1*nTRaEa_ZlXcFKXT-rxQHIg.png)

![img](https://miro.medium.com/max/1400/0*arP2ZshqOKgXss-h.png)







