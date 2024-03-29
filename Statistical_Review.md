# Statistical Review

References: 

[Applied Statistics and Probability for Engineers](https://www.goodreads.com/book/show/856620)



1. Probability Distributions

   - Discrete Distributions

     - Binomial Distribution, Bernoulli Distribution, Poisson Distribution

   - Continuous Distributions

     - Uniform Distribution, Exponential Distribution, Normal Distribution, Multivariate Normal

   - Inverse CFD method

     

2. Descriptive Statistics and Basic EDA

   - Some Definitions
     - Mean, Median, Standard Deviation, Standard Deviation, Standard Error
   - Basic EDA
     - Q-Q Plot
   - Statistical Intervals (Confident Intervals)
     - Normal distribution, variance known and unknown
     - chi-squared distribution, and promotion distribution, and bootstrap.

   Note that Statistical Inference has two parts: Estimation (point estimation, interval estimation, eta), and Hypothesis Tests. 

   

3. Hypothesis Testing 

   - Steps, Errors and Estimate (p-value), Two Errors, Significant Level and Statistical Power, Test Statistic Distributions
   - **Z-Test**: Confidence Interval, One-sample and Two-sample Z-tests, Pooled and Un-pooled two-sample Z-tests
   - **T-Test**: One-sample and two-sample T-tests, Pooled and Un-pooled T-tests
   - **Chi-Squared Tests**: Chi-squared test of association/Goodness-of-fit, test for categorical data, Test of variance
   - Kolmogorov-Smirnov test

   

4. Central Limit Theorem
   

5. Linear Regression

   - Maximum Likelihood Estimation (MLE)

   - Assumptions of Linear Regression: four assumption. 

   - Similar Linear Regression, MSE, R squared

   - Regression test: t-test, F-test

   - Other regression: Poisson regression

     

6. ANOVA

   - One-way ANOVA
   - Two-way ANOVA

   

7. A/B Testing

   - Types of Metrics (CTR, etc), Marginal Error
   - Binomial as a case study, determine the size of test
     

8. Bayesian Inference

   - Baye's Theorem

   - Prior and Posterior Distribution
   - 



## 0. Introduction



![img](http://staff.ustc.edu.cn/~zwp/teach/Prob-Stat/stat-rev.jpg)

![img](http://staff.ustc.edu.cn/~zwp/teach/Prob-Stat/stat-f.jpg)

[picture reference](http://staff.ustc.edu.cn/~zwp/teach/Prob-Stat/probstat.htm)



## 1. Probability Distributions

### 1.1 **Discrete Distributions**

- Binomial Distribution

  ![{\displaystyle f(k,n,p)=\Pr(k;n,p)=\Pr(X=k)={\binom {n}{k}}p^{k}(1-p)^{n-k}}](https://wikimedia.org/api/rest_v1/media/math/render/svg/b872c2c7bfaa26b16e8a82beaf72061b48daaf8e)

  ![Probability mass function for the binomial distribution](https://upload.wikimedia.org/wikipedia/commons/thumb/7/75/Binomial_distribution_pmf.svg/300px-Binomial_distribution_pmf.svg.png)

- Bernoulli Distribution

  ![{\displaystyle f(k;p)={\begin{cases}p&{\text{if }}k=1,\\q=1-p&{\text{if }}k=0.\end{cases}}}](https://wikimedia.org/api/rest_v1/media/math/render/svg/5fff3412509e73816dc2b28405b93c34f89ee487)

  

- Poisson Distribution

  ![\!f(k; \lambda)= \Pr(X{=}k)= \frac{\lambda^k e^{-\lambda}}{k!},](https://wikimedia.org/api/rest_v1/media/math/render/svg/6c429d187b5d4ef8ddea32a2d224f423cf9fe5b0)

  ![Poisson pmf.svg](https://upload.wikimedia.org/wikipedia/commons/thumb/1/16/Poisson_pmf.svg/325px-Poisson_pmf.svg.png)



### 1.2 **Continuous Distributions**

- Uniform Distribution

  ![f(x)={\begin{cases}{\frac {1}{b-a}}&\mathrm {for} \ a\leq x\leq b,\\[8pt]0&\mathrm {for} \ x<a\ \mathrm {or} \ x>b\end{cases}}](https://wikimedia.org/api/rest_v1/media/math/render/svg/b701524dbfea89ed90316dbc48c5b62954d7411c)

  ![PDF of the uniform probability distribution using the maximum convention at the transition points.](https://upload.wikimedia.org/wikipedia/commons/thumb/9/96/Uniform_Distribution_PDF_SVG.svg/250px-Uniform_Distribution_PDF_SVG.svg.png)

- Exponential Distribution

  ![{\displaystyle f(x;\lambda )={\begin{cases}\lambda e^{-\lambda x}&x\geq 0,\\0&x<0.\end{cases}}}](https://wikimedia.org/api/rest_v1/media/math/render/svg/a693ce9cd1fcd15b0732ff5c5b8040c359cc9332)

- Normal Distribution

  ![{\displaystyle f(x)={\frac {1}{\sigma {\sqrt {2\pi }}}}e^{-{\frac {1}{2}}\left({\frac {x-\mu }{\sigma }}\right)^{2}}}](https://wikimedia.org/api/rest_v1/media/math/render/svg/00cb9b2c9b866378626bcfa45c86a6de2f2b2e40)

![Normal Distribution PDF.svg](https://upload.wikimedia.org/wikipedia/commons/thumb/7/74/Normal_Distribution_PDF.svg/2880px-Normal_Distribution_PDF.svg.png)

1.3 **Multivariate Normal**

![{\displaystyle f_{\mathbf {X} }(x_{1},\ldots ,x_{k})={\frac {\exp \left(-{\frac {1}{2}}({\mathbf {x} }-{\boldsymbol {\mu }})^{\mathrm {T} }{\boldsymbol {\Sigma }}^{-1}({\mathbf {x} }-{\boldsymbol {\mu }})\right)}{\sqrt {(2\pi )^{k}|{\boldsymbol {\Sigma }}|}}}}](https://wikimedia.org/api/rest_v1/media/math/render/svg/c66e6f6abd66698181e114a4b00da97446efd3c4)

![img](https://upload.wikimedia.org/wikipedia/commons/thumb/5/57/Multivariate_Gaussian.png/300px-Multivariate_Gaussian.png)

Bivariate Case:

![{\displaystyle f(x,y)={\frac {1}{2\pi \sigma _{X}\sigma _{Y}{\sqrt {1-\rho ^{2}}}}}\exp \left(-{\frac {1}{2(1-\rho ^{2})}}\left[\left({\frac {x-\mu _{X}}{\sigma _{X}}}\right)^{2}-2\rho \left({\frac {x-\mu _{X}}{\sigma _{X}}}\right)\left({\frac {y-\mu _{Y}}{\sigma _{Y}}}\right)+\left({\frac {y-\mu _{Y}}{\sigma _{Y}}}\right)^{2}\right]\right)}](https://wikimedia.org/api/rest_v1/media/math/render/svg/c616921276f29c0c0cd5383fd81045939f8f6e82)

![{\boldsymbol  \mu }={\begin{pmatrix}\mu _{X}\\\mu _{Y}\end{pmatrix}},\quad {\boldsymbol  \Sigma }={\begin{pmatrix}\sigma _{X}^{2}&\rho \sigma _{X}\sigma _{Y}\\\rho \sigma _{X}\sigma _{Y}&\sigma _{Y}^{2}\end{pmatrix}}.](https://wikimedia.org/api/rest_v1/media/math/render/svg/1d6238c86bf561c952e0560e6f6ad3591278fb82)



### 1.3 Inverse CDF Method

CDF: cumulative distribution function

Question: [Why is the CDF of a sample uniformly distributed](https://stats.stackexchange.com/questions/161635/why-is-the-cdf-of-a-sample-uniformly-distributed)

![img](https://miro.medium.com/max/1364/1*WlFqo-hHs-pwDQa2_of0Xw.png)

Above picture see [this link](https://towardsdatascience.com/generate-random-variable-using-inverse-transform-method-in-python-8e5392f170a3)

For example, for exponential distribution: 

![img](https://miro.medium.com/max/1198/1*sUnvH5FPm-HL9dAHibrYBA@2x.png)



![img](https://miro.medium.com/max/654/1*HnHuldiEJ5UulWu2XZ-Lgw@2x.png)

![img](https://miro.medium.com/max/1002/1*0uEtsn-53sfIVqSXECv_xw@2x.png)

The code is:

```python
def exponential_inverse_trans(n=1,mean=1):
    U=np.random.uniform(0,1, size=n)
    X=-mean*np.log(1-U)
    actual=np.random.exponential(scale = mean, size=n)
    
    plt.figure(figsize=(12,9))
    plt.hist(X, bins=50, alpha=0.5, label="Generated r.v.")
    plt.hist(actual, bins=50, alpha=0.5, label="Actual r.v.")
    plt.title("Generated vs Actual %i Exponential Random Variables" %n)
    plt.legend()
    plt.show()
    return X
```



```python
cont_example2=exponential_inverse_trans(n=500,mean=3)
cont_example3=exponential_inverse_trans(n=10000,mean=3)
```

gives

![inverse_CDF_1.png](https://github.com/dongzhang84/Study_Notes/blob/main/figures/statistics/inverse_CDF_1.png?raw=true)

![inverse_CDF_2.png](https://github.com/dongzhang84/Study_Notes/blob/main/figures/statistics/inverse_CDF_2.png?raw=true)





## 2. Descriptive Statistics

### 2.1 Some Definitions

#### Sample Mean

![mean.png](https://github.com/dongzhang84/Study_Notes/blob/main/figures/statistics/mean.png?raw=true)

#### Standard Deviation

Discrete: 

![{\displaystyle \sigma ={\sqrt {{\frac {1}{N}}\sum _{i=1}^{N}(x_{i}-\mu )^{2}}},{\text{ where }}\mu ={\frac {1}{N}}\sum _{i=1}^{N}x_{i}.}](https://wikimedia.org/api/rest_v1/media/math/render/svg/98f02417b7c2830d941364f6b40e22ea63a9dd1f)

Continuous: 

![{\displaystyle {\begin{aligned}\sigma &\equiv {\sqrt {\operatorname {E} \left[(X-\mu )^{2}\right]}}={\sqrt {\int _{-\infty }^{+\infty }(x-\mu )^{2}f(x)dx}},\end{aligned}}}](https://wikimedia.org/api/rest_v1/media/math/render/svg/50fc45dec3e6e45e6b06bb50ea4d218269049d94)

```python
import numpy as np
a = np.array([[1, 2], [3, 4], [5,6], [7,8]])
np.std(a)
```

Return 

```
2.29128784747792
```

```python
>>> np.std(a, axis=0)
>>> array([2.23606798, 2.23606798])

>>> np.std(a, axis=1)
>>> array([0.5, 0.5, 0.5, 0.5])
```

where **axis = 0** means SD along the column and **axis = 1** means SD along the row.

Note that **Variance = std square**.



#### Sample Variance

The biased sample variance is then written:

![s_n^2 = \frac {1}{n} \sum_{i=1}^n  \left(x_i - \overline{x} \right)^ 2 = \frac{\sum_{i=1}^n \left(x_i^2\right)}{n} - \frac{\left(\sum_{i=1}^n x_i\right)^2}{n^2}](https://wikimedia.org/api/rest_v1/media/math/render/svg/1725a59716f931fd8dedf2c2bfc7d1cc6f02b566)

and the unbiased sample variance is written:

![s^2 = \frac {1}{n-1} \sum_{i=1}^n  \left(x_i - \overline{x} \right)^ 2 = \frac{\sum_{i=1}^n \left(x_i^2\right)}{n-1} - \frac{\left(\sum_{i=1}^n x_i\right)^2}{(n-1)n} = \left(\frac{n}{n-1}\right)\,s_n^2.](https://wikimedia.org/api/rest_v1/media/math/render/svg/6c61f055ff76396f4b98c926f2e3b2c47e7f64f0)

This is called [Bessel's correction](https://en.wikipedia.org/wiki/Bessel%27s_correction). The reason is 

![{\displaystyle {\begin{aligned}\operatorname {E} [S^{2}]&=\operatorname {E} {\bigg [}{\frac {1}{n}}\sum _{i=1}^{n}(X_{i}-\mu )^{2}-{\frac {2}{n}}({\overline {X}}-\mu )\sum _{i=1}^{n}(X_{i}-\mu )+({\overline {X}}-\mu )^{2}{\bigg ]}\\[8pt]&=\operatorname {E} {\bigg [}{\frac {1}{n}}\sum _{i=1}^{n}(X_{i}-\mu )^{2}-{\frac {2}{n}}({\overline {X}}-\mu )\cdot n\cdot ({\overline {X}}-\mu )+({\overline {X}}-\mu )^{2}{\bigg ]}\\[8pt]&=\operatorname {E} {\bigg [}{\frac {1}{n}}\sum _{i=1}^{n}(X_{i}-\mu )^{2}-2({\overline {X}}-\mu )^{2}+({\overline {X}}-\mu )^{2}{\bigg ]}\\[8pt]&=\operatorname {E} {\bigg [}{\frac {1}{n}}\sum _{i=1}^{n}(X_{i}-\mu )^{2}-({\overline {X}}-\mu )^{2}{\bigg ]}\\[8pt]&=\operatorname {E} {\bigg [}{\frac {1}{n}}\sum _{i=1}^{n}(X_{i}-\mu )^{2}{\bigg ]}-\operatorname {E} {\bigg [}({\overline {X}}-\mu )^{2}{\bigg ]}\\[8pt]&=\sigma ^{2}-\operatorname {E} {\bigg [}({\overline {X}}-\mu )^{2}{\bigg ]}=\left(1-{\frac {1}{n}}\right)\sigma ^{2}<\sigma ^{2}.\end{aligned}}}](https://wikimedia.org/api/rest_v1/media/math/render/svg/3f77971bd3f08e0c66f208221059b6978ab507e6)





#### Standard Error

![{\displaystyle {\sigma }_{\bar {x}}\ ={\frac {\sigma }{\sqrt {n}}}}](https://wikimedia.org/api/rest_v1/media/math/render/svg/f9dac77577c2717cbb973388e4d6563915705742)

![SE.png](https://github.com/dongzhang84/Study_Notes/blob/main/figures/statistics/SE.png?raw=true)



#### Standard Error of Mean and Median

**SE (median) = 1.2533 × SE( )**, [reference](https://influentialpoints.com/Training/standard_error_of_median)

```python
sigma=np.std(data)
n=len(data)
sigma_median=1.253*sigma/np.sqrt(n)
```





### 2.2 Basic EDA

#### 2.2.1 Q-Q Plot

A Q–Q (quantile-quantile) plot is a probability plot, which is a graphical method for comparing two probability distributions by plotting their quantiles against each other.

For example:

![img](https://upload.wikimedia.org/wikipedia/commons/thumb/0/08/Normal_normal_qq.svg/300px-Normal_normal_qq.svg.png)



#### 2.2.2 Bootstrapping

Regarding Boostrapping resampling size: https://tools4dev.org/resources/how-to-choose-a-sample-size/





### 2.3 Confidence Intervals

The confidence interval (CI) is a range of values that’s likely to include a population value with a certain degree of confidence. It is often expressed as a % whereby a population mean lies between an upper and lower interval. 



![CI = \bar{x} \pm z \frac{s}{\sqrt{n}}](https://www.gstatic.com/education/formulas2/397133473/en/confidence_interval_formula.svg)









![img](https://upload.wikimedia.org/wikipedia/commons/b/bb/Normal_distribution_and_scales.gif)



- | C    | z*    |
  | ---- | ----- |
  | 99%  | 2.576 |
  | 98%  | 2.326 |
  | 95%  | 1.96  |
  | 90%  | 1.645 |





#### 2.3.1 Normal Distribution, Variance Known

![CI_1.png](https://github.com/dongzhang84/Study_Notes/blob/main/figures/statistics/CI_1.png?raw=true)



**Choice of Sample Size**

![CI_2.png](https://github.com/dongzhang84/Study_Notes/blob/main/figures/statistics/CI_2.png?raw=true)



**One-Sided Confidence Bounds**

![CI_3.png](https://github.com/dongzhang84/Study_Notes/blob/main/figures/statistics/CI_3.png?raw=true)



**Large-Sample Confidence Interval for μ**

![CI_4.png](https://github.com/dongzhang84/Study_Notes/blob/main/figures/statistics/CI_4.png?raw=true)



#### 2.3.2. Normal Distribution, Variance Unknown





***t* Confidence Interval **

![CI_5.png](https://github.com/dongzhang84/Study_Notes/blob/main/figures/statistics/CI_5.png?raw=true)



#### 2.3.3. Variance and Standard Deviation of a Normal Distribution

**chi-squared distribution**

![CI_6.png](https://github.com/dongzhang84/Study_Notes/blob/main/figures/statistics/CI_6.png?raw=true)



![CI_7.png](https://github.com/dongzhang84/Study_Notes/blob/main/figures/statistics/CI_7.png?raw=true)



#### 2.3.4 Confidence Interval for a Population Proportion

![CI_8.png](https://github.com/dongzhang84/Study_Notes/blob/main/figures/statistics/CI_8.png?raw=true)

![CI_9.png](https://github.com/dongzhang84/Study_Notes/blob/main/figures/statistics/CI_9.png?raw=true)

![CI_10.png](https://github.com/dongzhang84/Study_Notes/blob/main/figures/statistics/CI_10.png?raw=true)



#### 2.3.5 Bootstrap Confidence Interval







## 3. Hypothesis Tests



A statistical hypothesis is a hypothesis that is testable on the basis of observed data modeled as the realized values taken by a collection of random variables. 

### 3.1) Steps

(From David Spiegelhalter's "The Art of Statistics")

1. Set up a question in terms of a **null hypothesis** that we want to check. This is generally given the notation H0.
2. Choose a **test statistic** that estimates something  that, if it turned out to be extreme enough, would lead us to double the null hypothesis. 
3. Generate the sampling distribution of this test statistic, were the **null hypothesis** true. 
4. Check whether our observed statistic lies in the tails of this distribution and summarize this by the **P-value**: the probability, were the null hypothesis true, of observing such an extreme statistic. The **P-value** is therefore a particular tail-area.
5. Declare the result statistically significant if the **P-value** is below some critical threshold.  



![img](https://miro.medium.com/max/1400/1*8pSgz0bAlIQ3wlGNJAc-6g.png)



### 3.2) Errors and Estimate

**P-value**

In null hypothesis significance testing, the p-value is the probability of obtaining test results at least as extreme as the results actually observed, under the assumption that the null hypothesis is correct.

- ![{\displaystyle p=\Pr(T\geq t\mid H_{0})}](https://wikimedia.org/api/rest_v1/media/math/render/svg/ea300b1ffc1728f5a10bc4ef20749c559d2802ba) for a one-sided right-tail test,
- ![{\displaystyle p=\Pr(T\leq t\mid H_{0})}](https://wikimedia.org/api/rest_v1/media/math/render/svg/c21b6377a769dd29cac292258b5a1d54bd4cc240) for a one-sided left-tail test,
- ![{\displaystyle p=2\min\{\Pr(T\geq t\mid H_{0}),\Pr(T\leq t\mid H_{0})\}}](https://wikimedia.org/api/rest_v1/media/math/render/svg/50ddf05b387d4b2456864cea94cd592e69ac3fd7) for a two-sided test. If distribution T is symmetric about zero, then ![{\displaystyle p=\Pr(|T|\geq |t|\mid H_{0})}](https://wikimedia.org/api/rest_v1/media/math/render/svg/711243ba83abe721ec22fadb229af1445a21bd92)

![P-value in statistical significance testing.svg](https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/P-value_in_statistical_significance_testing.svg/370px-P-value_in_statistical_significance_testing.svg.png)

**Two Errors**

- **Type I error**: a type I error is the mistaken rejection of the null hypothesis (probability: $\alpha$).
- **Type II error**: a type II error is the mistaken acceptance of the null hypothesis (probability: $\beta$).

**Significant Level**: $\alpha$

**Statistical Power**: 1- $\beta$



#### 3.2.1) Test Statistic Distribution

In standard cases this will be a well-known result. For example,

- [student's t-distribution](https://en.wikipedia.org/wiki/Student%27s_t-distribution) (known degrees of freedom)
- [normal distribution](https://en.wikipedia.org/wiki/Normal_distribution) (known mean and variance)



### 3.3) Various Tests

----

#### 3.3.1. Z-test

A Z-test is any statistical test for which the distribution of the test statistic under the null hypothesis can be approximated by a **normal distribution**.



**1.a) One Sample Z-test**

Samples come from a same population

![z_test1.png](https://github.com/dongzhang84/Study_Notes/blob/main/figures/statistics/z_test1.png?raw=true)

- **Step 1**: State the Null Hypothesis. 

- **Step 2**: Use the z-formula to find a z-score.
  ![One Sample Z Test](https://www.statisticshowto.com/wp-content/uploads/2014/02/z-score-formula.jpg)

- **Step 3**: Compute P-value.
  For example, Tabled value for *z* ≤ 1.28 is 0.8997

  P-value is 1 – 0.8997 = 0.1003

- **Step 4**: Evaluate null hypothesis. 

![Two Sample Z Test of Proportions](https://sixsigmastudyguide.com/wp-content/uploads/2021/01/zp7-1.png)



##### 1.b) **Two Sample Z-test** [reference](https://sixsigmastudyguide.com/two-sample-test-of-proportions/)

**Requirements**: Two normally distributed but independent populations, σ is known.

![Two Sample Z Test of Proportions](https://sixsigmastudyguide.com/wp-content/uploads/2021/01/zp1.png)

**1.c) Binomial Example**

For np > 10, binomial can be considered as normal distribution. 

- **Pooled Z test of proportions formula**

![img](https://sixsigmastudyguide.com/wp-content/uploads/2021/01/zp2-1.png)

where ![The one and two sample proportion hypothesis ](https://sixsigmastudyguide.com/wp-content/uploads/2019/10/on7.png)

- **Un-pooled Z test of proportions formula**

![Two Sample Z Test of Proportions](https://sixsigmastudyguide.com/wp-content/uploads/2021/01/zp3-1.png)

**Z** – ![img](https://www.analystsoft.com/en/products/statplus/content/help/analysis_basic_statistics_two_sample_z-test_files/image004.png) critical value for z.

**P(Z<=z)  p-level** – probability of observing the sample statistic as extreme as the test statistic. 



**1.4) Unpair Z-test**

The unpaired Z-test statistic is

![{\frac  {{\bar  {Y}}_{2}-{\bar  {Y}}_{1}}{{\sqrt  {\sigma _{1}^{2}/n+\sigma _{2}^{2}/n}}}},](https://wikimedia.org/api/rest_v1/media/math/render/svg/7f01b62a7c57f1bf513343ed62b856aa0663dfe8)



------

#### 3.3.2. T-test

The t-test is any statistical hypothesis test in which the test statistic follows a **Student's t-distribution** under the null hypothesis.



2.a **One Sample and Two Sample T-tests**:

- A **one-sample**  location test of whether the mean of a population has a value specified in a null hypothesis. 
  Determines whether the sample mean is statistically different from a known or hypothesized population mean. The One Sample T-test is a parametric test.

  ![formula to calculate t for a 1-sample t-test](https://blog.minitab.com/hubfs/Imported_Blog_Media/formula_1t.png)

  By the [central limit theorem](https://en.wikipedia.org/wiki/Central_limit_theorem), if the observations are independent and the second moment exists, then t will be approximately normal N(0;1).

  Code:

  ```python
  scipy.stats.ttest_1samp
  from scipy import stats
  
  np.random.seed(7654567)  # fix seed to get the same result
  rvs = stats.norm.rvs(loc=5, scale=10, size=(50,2))
  
  stats.ttest_1samp(rvs,5.0)
  (array([-0.68014479, -0.04323899]), array([ 0.49961383,  0.96568674]))
  stats.ttest_1samp(rvs,0.0)
  (array([ 2.77025808,  4.11038784]), array([ 0.00789095,  0.00014999]))
  ```

  

- A **two-sample** location test of the **null hypothesis** such that **the means of two populations are equal**.

  See two-sample T-test [example](https://www.jmp.com/en_us/statistics-knowledge-portal/t-test/two-sample-t-test.html).

  

2.b **Unpaired and Paired two-sample T-tests** 

Two-sample *t*-tests for a difference in mean involve **independent samples (unpaired samples)** or paired samples.

2.c **Correlated (or Paired) T-Test**

This sort-of-paired observation happens when there is no presumption that all of the measurements are from the same population, and when *n* is the same for both samples.

![t_test_2sample_paired.png](https://github.com/dongzhang84/Study_Notes/blob/main/figures/statistics/t_test_2sample_paired.png?raw=true)



2.d **Equal Variance (or Pooled) T-Test**

![{\displaystyle t={\frac {{\bar {X}}_{1}-{\bar {X}}_{2}}{s_{p}\cdot {\sqrt {{\frac {1}{n_{1}}}+{\frac {1}{n_{2}}}}}}}}](https://wikimedia.org/api/rest_v1/media/math/render/svg/faf70034d0a3a686080b98b32f64f2cc62a5dbad)

where 

![{\displaystyle s_{p}={\sqrt {\frac {\left(n_{1}-1\right)s_{X_{1}}^{2}+\left(n_{2}-1\right)s_{X_{2}}^{2}}{n_{1}+n_{2}-2}}}}](https://wikimedia.org/api/rest_v1/media/math/render/svg/451cd1fdd5cb308af96957c3cb28131035b07e97)

is an estimator of the pooled standard deviation of the two samples.

Degree of freedom: n1+n2-2



2.e **Unequal Variance T-Test**

See [Welch's t-test](https://en.wikipedia.org/wiki/Welch%27s_t-test)

![{\displaystyle t={\frac {{\bar {X}}_{1}-{\bar {X}}_{2}}{s_{\bar {\Delta }}}}}](https://wikimedia.org/api/rest_v1/media/math/render/svg/eeecf40c622f1fa6b0fb9462c7c4b7030cbb47eb)

where

![{\displaystyle s_{\bar {\Delta }}={\sqrt {{\frac {s_{1}^{2}}{n_{1}}}+{\frac {s_{2}^{2}}{n_{2}}}}}.}](https://wikimedia.org/api/rest_v1/media/math/render/svg/5024dd38e905d60353a324af2a05058fdd4ac3e7)

and the degree of freedom (d.f.) can be calculated by

![{\displaystyle \mathrm {d.f.} ={\frac {\left({\frac {s_{1}^{2}}{n_{1}}}+{\frac {s_{2}^{2}}{n_{2}}}\right)^{2}}{{\frac {\left(s_{1}^{2}/n_{1}\right)^{2}}{n_{1}-1}}+{\frac {\left(s_{2}^{2}/n_{2}\right)^{2}}{n_{2}-1}}}}.}](https://wikimedia.org/api/rest_v1/media/math/render/svg/bf9929fa7098ff7e57a5c421c0fdcb3eea20435f)



Demonstration

![T-test](https://www.investopedia.com/thmb/J-NC9PcSiu_SrtLk59nYeJqHQFA=/6250x4623/filters:no_upscale():max_bytes(150000):strip_icc():format(webp)/ttest22-0afd4aefe9cc42628f603dc2c7c5f69a.png)



----

#### 3.3.3. Chi-Squared Tests

3.a **Chi-Squared Test of Association/Goodness-of-Fit Test** 

The Chi-square goodness of fit test is **a statistical hypothesis test used to determine whether a variable is likely to come from a specified distribution or not**.

Non-normal, and non-t distribution. But an assumed model distribution. 

A statistical test that indicates the degree of incompatibility of data with an assumed statistical model comprising the null hypothesis, which may be one of lack of association, or some other specified mathematical form. Specifically, the test compares a set of $m$ Observed counts $o_1, o_2, o_3, o_4 ... o_m$ With a set of expected values $e_1, e_2, e_3, ... e_m$, which have been calculated under the null hypothesis. The simplest version of the test statistic is given as
$$
\begin{equation}
X^2 = \sum_{j=1}^{m} \frac{(o_j - e_j)^2}{e_j}
\end{equation}
$$
Under the null hypothesis $X^2$ Will have an approximate chi-square sampling distribution, enabling an associated  **P-value** to be calculated. 



**Example: Compare Number of Real data to Expected Numbers**

The following [example](https://www.jmp.com/en_ch/statistics-knowledge-portal/chi-square-test/chi-square-goodness-of-fit-test.html):

![chi_squred_2.png](https://github.com/dongzhang84/Study_Notes/blob/main/figures/statistics/chi_squred_2.png?raw=true)

![img](https://www.jmp.com/en_ch/statistics-knowledge-portal/chi-square-test/chi-square-goodness-of-fit-test/_jcr_content/par/styledcontainer_2069/par/image_523413870.img.png/1623884285244.png)





The following [example](https://en.wikipedia.org/wiki/Chi-squared_test#Chi-squared_test_for_variance_in_a_normal_population):

**Example chi-squared test for categorical data**

![chi_squred_3.png](https://github.com/dongzhang84/Study_Notes/blob/main/figures/statistics/chi_squred_3.png?raw=true)



3.b **Chi-Squared Test of Variance**

A chi-square test ( [Snedecor and Cochran, 1983](https://www.itl.nist.gov/div898/handbook/eda/section4/eda43.htm#Snedecor)) can be used to test if the variance of a population is equal to a specified value. This test can be either a two-sided test or a one-sided test. 

[Reference](https://www.itl.nist.gov/div898/handbook/eda/section3/eda358.htm)

![chi_squred_4.png](https://github.com/dongzhang84/Study_Notes/blob/main/figures/statistics/chi_squred_4.png?raw=true)

![chi_squred_5.png](https://github.com/dongzhang84/Study_Notes/blob/main/figures/statistics/chi_squred_5.png?raw=true)



### 3.4) Kolmogorov-Smirnov test

![img](https://upload.wikimedia.org/wikipedia/commons/thumb/c/cf/KS_Example.png/300px-KS_Example.png)

**Goodness-of-fit**: Illustration of the Kolmogorov–Smirnov statistic. The red line is a model CDF, the blue line is an empirical CDF, and the black arrow is the K–S statistic.

https://www.tutorialspoint.com/statistics/kolmogorov_smirnov_test.htm





### 3.5）Bonferroni correction

The Bonferroni correction is a method to counteract the problem of multiple comparisons.

The [familywise error rate](https://en.wikipedia.org/wiki/Familywise_error_rate) (FWER): ![{\displaystyle {\text{FWER}}=P\left\{\bigcup _{i=1}^{m_{0}}\left(p_{i}\leq {\frac {\alpha }{m}}\right)\right\}\leq \sum _{i=1}^{m_{0}}\left\{P\left(p_{i}\leq {\frac {\alpha }{m}}\right)\right\}=m_{0}{\frac {\alpha }{m}}\leq \alpha .}](https://wikimedia.org/api/rest_v1/media/math/render/svg/f5a25e29ec5478b06b9856a892469c72ac50b285)





## 4. Central Limited Theorem

#### 4.1) The Law of Large Numbers

It is a theorem that describes the result of performing the same experiment a large number of times. According to the law, the **average** of the results obtained from a large number of trials should be close to the expected value and will tend to become closer to the **expected value** as more trials are performed. 



**Weak Law**

![\lim _{n\to \infty }\Pr \!\left(\,|{\overline {X}}_{n}-\mu |>\varepsilon \,\right)=0.](https://wikimedia.org/api/rest_v1/media/math/render/svg/e3848a0ff097c73716a0bfc4df59c18691b2a323)

**Strong Law** 

![\Pr \!\left(\lim _{n\to \infty }{\overline {X}}_{n}=\mu \right)=1.](https://wikimedia.org/api/rest_v1/media/math/render/svg/befeda3c4b77efb2cf7835a9569edaadebd978e9)



#### 4.2) Central Limited Theorem

![image-20211020210955369](/Users/dong/Library/Application Support/typora-user-images/image-20211020210955369.png)



If $X_1, X_2, X_3, ..., X_n$ are $n$ random samples drawn from a population with overall mean $\mu$ And finite variance $\sigma^2$, and if $\bar{X}_n$ Is the sample mean, then the limiting form of the distribution,

 ![{\textstyle Z=\lim _{n\to \infty }{\sqrt {n}}{\left({\frac {{\bar {X}}_{n}-\mu }{\sigma }}\right)}}](https://wikimedia.org/api/rest_v1/media/math/render/svg/d269e67503670688dba6d8c25481f5e63c0b1d5b), 

is a standard normal distribution.

(Need to expand more)



## 5. Linear Regression and other Regressions

### 5.1) Maximum Likelihood Estimation

Maximum likelihood estimation (MLE) is a method of estimating the parameters of an assumed probability distribution. 

Evaluating the joint density at the observed data sample:

![{\displaystyle L_{n}(\theta )=L_{n}(\theta ;\mathbf {y} )=f_{n}(\mathbf {y} ;\theta )}](https://wikimedia.org/api/rest_v1/media/math/render/svg/fa485e7acf98b3a0ce236ce7293f63dd89f84b96)

The goal of maximum likelihood estimation is to find the values of the model parameters that maximize the likelihood function over the parameter space, that is

![{\displaystyle {\hat {\theta }}={\underset {\theta \in \Theta }{\operatorname {arg\;max} }}\,{\widehat {L}}_{n}(\theta \,;\mathbf {y} )}](https://wikimedia.org/api/rest_v1/media/math/render/svg/266fa29a1e5c9905225936633538f30e3db04529)

In practice, it is often convenient to work with the natural logarithm of the likelihood function, called the log-likelihood:

![{\displaystyle \ell (\theta \,;\mathbf {y} )=\ln L_{n}(\theta \,;\mathbf {y} ).}](https://wikimedia.org/api/rest_v1/media/math/render/svg/70323a65c0f24cb9b3e9bb0e1a8cf30442c350a7)

![{\displaystyle {\frac {\partial \ell }{\partial \theta _{1}}}=0,\quad {\frac {\partial \ell }{\partial \theta _{2}}}=0,\quad \ldots ,\quad {\frac {\partial \ell }{\partial \theta _{k}}}=0,}](https://wikimedia.org/api/rest_v1/media/math/render/svg/ee2c70f8b76d0ba39fa9fa7a319027851566147e)



### 5.2) Assumptions of Linear Regression



**Linear relationship:** There exists a linear relationship between the independent variable, x, and the 

**Independence:** The residuals are independent. In particular, there is no correlation between consecutive residuals in time series data.

**Normality**：The residuals of the model are normally distributed.

**Homoscedasticity:** The residuals have constant variance at every level of x.



Another version: 



**1. Linearity:** The relationship between  X and  Y must be linear.

Check this assumption by examining a scatterplot of x and y.

**2. Independence of errors:** There is not a relationship between the residuals and the Y  variable; in other words,  Y is independent of errors.

Check this assumption by examining a scatterplot of “residuals versus fits”; the correlation should be approximately 0. In other words, there should not look like there is a relationship.

**3. Normality of errors:** The residuals must be approximately normally distributed.

Check this assumption by examining a normal probability plot; the observations should be near the line. You can also examine a histogram of the residuals; it should be approximately normally distributed.

**4. Equal variances:** The variance of the residuals is the same for all values of X.

Check this assumption by examining the scatterplot of “residuals versus fits”; the variance of the residuals should be the same across all values of the x-axis. If the plot shows a pattern (e.g., bowtie or megaphone shape), then variances are not consistent, and this assumption has not been met.



For example: 

![img](https://upload.wikimedia.org/wikipedia/commons/9/93/Homoscedasticity.png)

Plot with random data showing homoscedasticity: at each value of x, the y-value of the dots has about the same variance.



### 5.3) Simple Linear Regression



Consider a model function: 

![y=\alpha +\beta x,](https://wikimedia.org/api/rest_v1/media/math/render/svg/bf2c1cac7c1e6c9a426d92e9adad6ff4d8b4152e)

We can describe the underlying relationship between $y_i$ and $x_i$  as

![ y_i = \alpha + \beta x_i + \varepsilon_i.](https://wikimedia.org/api/rest_v1/media/math/render/svg/968be557dd22b1a2e536b8d22369cfdb37f58703)

And the solution of the parameters are

![{\textstyle {\begin{aligned}{\widehat {\alpha }}&={\bar {y}}-({\widehat {\beta }}\,{\bar {x}}),\\[5pt]{\widehat {\beta }}&={\frac {\sum _{i=1}^{n}(x_{i}-{\bar {x}})(y_{i}-{\bar {y}})}{\sum _{i=1}^{n}(x_{i}-{\bar {x}})^{2}}}\\[6pt]&={\frac {s_{x,y}}{s_{x}^{2}}}\\[5pt]&=r_{xy}{\frac {s_{y}}{s_{x}}}.\\[6pt]\end{aligned}}}](https://wikimedia.org/api/rest_v1/media/math/render/svg/9caed0f59417a425c988764032e5892130e97fa4)

Note that 

![{\displaystyle {\begin{aligned}{\widehat {\beta }}&={\frac {\sum _{i=1}^{n}(x_{i}-{\bar {x}})(y_{i}-{\bar {y}})}{\sum _{i=1}^{n}(x_{i}-{\bar {x}})^{2}}}={\frac {\sum _{i=1}^{n}(x_{i}-{\bar {x}})^{2}*{\frac {(y_{i}-{\bar {y}})}{(x_{i}-{\bar {x}})}}}{\sum _{i=1}^{n}(x_{i}-{\bar {x}})^{2}}}\\[6pt]\end{aligned}}}](https://wikimedia.org/api/rest_v1/media/math/render/svg/e1ac1b7ef40d1c91a192327f20ae7ca88f4c4d37)



### 5.4) Metrics of model evaluation

See [reference here](https://vitalflux.com/mean-square-error-r-squared-which-one-to-use/).

**Mean Squared Error (MSE)**

<img src="https://vitalflux.com/wp-content/uploads/2020/09/Screenshot-2020-09-30-at-2.58.15-PM.png" alt="Mean squared error" style="zoom: 33%;" />

![Mean Squared Error Representation](https://vitalflux.com/wp-content/uploads/2020/09/Regression-terminologies-Page-2-1024x619.png)



**R-squared**

![Digrammatic representation for understanding R-Squared](https://vitalflux.com/wp-content/uploads/2020/09/Regression-terminologies-Page-3.png)

![R-Squared as ration of SSR and SST](https://vitalflux.com/wp-content/uploads/2020/09/Screenshot-2020-09-30-at-3.57.00-PM.png)

![R-Squared as a function of MSE](https://vitalflux.com/wp-content/uploads/2020/09/Screenshot-2020-09-30-at-6.07.51-PM.png)

R-squared and Adjust R-squared

![R_squared.png](https://github.com/dongzhang84/Study_Notes/blob/main/figures/statistics/R_squared.png?raw=true)



### 5.5) More general formula

[reference](https://www.cnblogs.com/itboys/p/8409590.html)



![{\displaystyle {\vec {\hat {\beta }}}={\underset {\vec {\beta }}{\mbox{arg min}}}\,L\left(D,{\vec {\beta }}\right)={\underset {\vec {\beta }}{\mbox{arg min}}}\sum _{i=1}^{n}\left({\vec {\beta }}\cdot {\vec {x_{i}}}-y_{i}\right)^{2}}](https://wikimedia.org/api/rest_v1/media/math/render/svg/e286875158615d5647351c312f4609a125a5d943)



![{\displaystyle {\begin{aligned}-2X^{\textsf {T}}Y+2X^{\textsf {T}}X{\vec {\beta }}&=0\\\Rightarrow X^{\textsf {T}}Y&=X^{\textsf {T}}X{\vec {\beta }}\\\Rightarrow {\vec {\hat {\beta }}}&=\left(X^{\textsf {T}}X\right)^{-1}X^{\textsf {T}}Y\end{aligned}}}](https://wikimedia.org/api/rest_v1/media/math/render/svg/fb161a08d2371663a73aae304a7ba910b7e9776a)



**Ridge:**

![img](https://images2017.cnblogs.com/blog/906988/201802/906988-20180203151204000-1881679862.png)

 

**Lasso**

![img](https://images2017.cnblogs.com/blog/906988/201802/906988-20180203151240781-770033962.png)







### 5.6) Regression test

Here is a very good reference on regression test: [Linear Regression: Test and Confidence Intervals](https://www2.isye.gatech.edu/~yxie77/isye2028/lecture12.pdf)

**T-Test**

![regress_t.png](https://github.com/dongzhang84/Study_Notes/blob/main/figures/statistics/regress_t.png?raw=true)

**F-Test**

![regress_f.png](https://github.com/dongzhang84/Study_Notes/blob/main/figures/statistics/regress_f.png?raw=true)

Some references to read

https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLSResults.t_test.html

https://stackoverflow.com/questions/27928275/find-p-value-significance-in-scikit-learn-linearregression



### 5.6) Other Regression: Poisson Regression

Poisson regression is a generalized linear model form of regression analysis used to model **count data** and **contingency tables**.

![{\displaystyle \lambda :=\operatorname {E} (Y\mid x)=e^{\theta 'x},\,}](https://wikimedia.org/api/rest_v1/media/math/render/svg/a0da7d4c02f2f34dfd6c08455f770437672d2b38)

![{\displaystyle p(y\mid x;\theta )={\frac {\lambda ^{y}}{y!}}e^{-\lambda }={\frac {e^{y\theta 'x}e^{-e^{\theta 'x}}}{y!}}}](https://wikimedia.org/api/rest_v1/media/math/render/svg/3698dbb249c32c1f555f620f6277ddf4d3bd4de1)

![p(y_{1},\ldots ,y_{m}\mid x_{1},\ldots ,x_{m};\theta )=\prod _{i=1}^{m}{\frac {e^{y_{i}\theta 'x_{i}}e^{-e^{\theta 'x_{i}}}}{y_{i}!}}.](https://wikimedia.org/api/rest_v1/media/math/render/svg/8df9dcc9459e1dfb07dfa3c781b301b60b60285e)

Likelihood function:

![{\displaystyle L(\theta \mid X,Y)=\prod _{i=1}^{m}{\frac {e^{y_{i}\theta 'x_{i}}e^{-e^{\theta 'x_{i}}}}{y_{i}!}}.}](https://wikimedia.org/api/rest_v1/media/math/render/svg/83bd2e605da72a452e7f3aad926ab6ccb255f269)



- Example 1. The number of persons killed by mule or horse kicks in the Prussian army per year. von Bortkiewicz collected data from 20 volumes of Preussischen Statistik. These data were collected on 10 corps of the Prussian army in the late 1800s over the course of 20 years.

- Example 2. A health-related researcher is studying the number of hospital visits in past 12 months by senior citizens in a community based on the characteristics of the individuals and the types of health plans under which each one is covered.
  
- Example 3. A researcher in education is interested in the association between the number of awards earned by students at one high school and the students’ performance in math and the type of program (e.g., vocational, general or academic) in which students were enrolled.







## 6. ANOVA (Analysis of Variance)

### 6.1) One-way ANOVA

The one-way analysis of variance (ANOVA) is used to determine whether there are any statistically significant differences between the means of three or more independent (unrelated) groups. 

![One-way ANOVA Null Hypothesis](https://statistics.laerd.com/statistical-guides/img/one-way-anova-null-hypothesis.png)

https://libguides.library.kent.edu/spss/onewayanova

The test statistic for a One-Way ANOVA is denoted as *F*. For an independent variable with *k* groups, the *F* statistic evaluates whether the group means are significantly different. Because the computation of the F statistic is slightly more involved than computing the paired or independent samples t test statistics, it's extremely common for all of the *F* statistic components to be depicted in a table like the following:

|           | Sum of Squares |   df | Mean Square |       F |
| :-------- | -------------: | ---: | ----------: | ------: |
| Treatment |            SSR |  dfr |         MSR | MSR/MSE |
| Error     |            SSE |  dfe |         MSE |         |
| Total     |            SST |  dfT |             |         |

See example here:

https://en.wikipedia.org/wiki/One-way_analysis_of_variance

Or from the textbook:

![anova_1.png](https://github.com/dongzhang84/Study_Notes/blob/main/figures/statistics/anova_1.png?raw=true)



![anova_2.png](https://github.com/dongzhang84/Study_Notes/blob/main/figures/statistics/anova_2.png?raw=true)

![anova_3.png](https://github.com/dongzhang84/Study_Notes/blob/main/figures/statistics/anova_3.png?raw=true)



### 6.2) Two-way ANOVA

A two-way ANOVA is used to estimate how the mean of a quantitative variable changes according to the levels of two different categorical independent variables. 

A two-way ANOVA with interaction tests three null hypotheses at the same time:

- There is no difference in group means at any level of the first independent variable.
- There is no difference in group means at any level of the second independent variable.
- The effect of one independent variable does not depend on the effect of the other independent variable (a.k.a. no interaction effect).

https://www.itl.nist.gov/div898/handbook/prc/section4/prc437.htm

![anova_4.png](https://github.com/dongzhang84/Study_Notes/blob/main/figures/statistics/anova_4.png?raw=true)

![anova_5.png](https://github.com/dongzhang84/Study_Notes/blob/main/figures/statistics/anova_5.png?raw=true)



See this example: https://www.statology.org/two-way-anova/



![anova_6.png](https://github.com/dongzhang84/Study_Notes/blob/main/figures/statistics/anova_6.png?raw=true)

https://www.youtube.com/watch?v=cNIIn9bConY





## 7. A/B Testing

A/B testing (also known as bucket testing or split-run testing) is a user experience research methodology. 

A/B tests consist of a randomized experiment with two variants, A and B. It includes application of statistical hypothesis testing or "two-sample hypothesis testing" as used in the field of statistics. A/B testing is a way to compare two versions of a single variable, typically by testing a subject's response to variant A against variant B, and determining which of the two variants is more effective.

![img](https://upload.wikimedia.org/wikipedia/commons/2/2e/A-B_testing_example.png)



AB test 一般的套路就是 [reference](https://www.moonbbs.com/thread-3748725-1-1.html)

- understand business goal
- define metrics
- hypo
- design test plans, sample size? Duration? Regions for AB testing? 
- launch experiment 
- sanity check and analyze result 
- conclusion/suggestion



**About Sanity Check**

http://napitupulu-jon.appspot.com/posts/sanity-check-abtesting-udacity.html



### Types of Metrics

- Click through rate = # clicks / # visits
- Click through probability = # unique visitors who click / # unique visitors
- Retention rate = # users remained at the end of evaluation period / # users at the beginning of the evaluation period
- Churn rate = # users churned / # total users at the beginning of the evaluation period
- ROI = (current value – investment) / investment



### Another Article to summarize Udacity A/B Testing course

https://towardsdatascience.com/a-summary-of-udacity-a-b-testing-course-9ecc32dedbb1

#### Step 1: Choose and characterize metrics for both sanity check and evaluation

**How to measure the sensitivity and robustness?**

- Run experiments
- Use A/A test to see if metrics pick up difference (if yes, then the metric is not robust)
- Retrospective analysis
- 

#### Step 2: Choose significance level, statistical power and practical significance level

Usually the significance level is 0.05 and power is set as 0.8. 



#### Step 3: Calculate required sample size

- **Subject**: What is the subject (**unit of diversion**) of the test? I.e. what are the units you are going to run the test on and comparing. Unit of diversion can be event based (e.g. pageview) or anonymous ID(e.g. cookie id) or user ID. These are commonly used unit of diversion. For user visible changes, you want to use user_id or cookie to measure the change. If measuring latency change, other metrics like event level diversion might be enough.

  

- **Population**: What subjects are eligible for the test? Everyone? Only people in the US? Only people in certain industry?

- **How to reduce the size of an experiment to get it done faster?** You can increase significance level alpha, or reduce power (1-beta) which means increase beta, or change the unit of diversion if originally it is not the same with unit of analysis (unit of analysis: denominator of your evaluation metric) .



#### Step 4: Take sample for control/treatment groups and run the test

Several things to keep in mind:

- **Duration**: What’s the best time to run it? Students going back to college? Holidays? Weekend vs. weekdays?
- **Exposure**: What fraction of traffic you want to expose the experiment to? Suggestion is take a small fraction, run multiple tests at the same time (different days: weekend, weekday, holiday).
- **Learning effect**: When there’s a new change, in the beginning users may against the change or use the change a lot. But overtime, user behavior becomes stable, which is called plateau stage. The key thing to measure learning effect is time, but in reality you don’t have that much luxury of taking that much time to make a decision. Suggestion: run on a smaller group of users, for a longer period of time.



#### Step 5: Analyze the results and draw conclusions

First step, sanity check.

Second step, analyze the results.

Last step, draw conclusions.



-------



**Marginal Error**

Marginal error (M) = Z score of the confidence interval * Standard Error (SE)

Marginal error is function of proportion of success and sample size – means need to consider the proportion of success when deciding the sample size.

![001.jpg](https://github.com/dongzhang84/Study_Notes/blob/main/figures/AB_testing/001.jpg?raw=true)



**Ideal A/B Testing Result**



See this [reference](https://towardsdatascience.com/bootstrapping-confidence-intervals-the-basics-b4f28156a8da):

![img](https://miro.medium.com/max/1400/0*5UBGh1SVV7ujYMjk.png)



### Binomial As an Example

Some basic equations/definitions:

![002.jpg](https://github.com/dongzhang84/Study_Notes/blob/main/figures/AB_testing/002.jpg?raw=true)

#### How to Carry out Hypothesis Testing

Pooled sample:

![003.jpg](https://github.com/dongzhang84/Study_Notes/blob/main/figures/AB_testing/003.jpg?raw=true)

![004.jpg](https://github.com/dongzhang84/Study_Notes/blob/main/figures/AB_testing/004.jpg?raw=true)

One example: 

![005.png](https://github.com/dongzhang84/Study_Notes/blob/main/figures/AB_testing/005.png?raw=true)

**Do we want to launch new features? ** 

![006.png](https://github.com/dongzhang84/Study_Notes/blob/main/figures/AB_testing/006.png?raw=true)



#### Given alpha and beta, decide Sample Size for testing

- **Significant level**: 
  alpha. **False Postive**.

- **Statistical Power**: 
  1- beta (beta is **False Negative**). The statistical **power** of a binary [hypothesis test](https://en.wikipedia.org/wiki/Hypothesis_test) is the probability that the test correctly rejects the [null hypothesis](https://en.wikipedia.org/wiki/Null_hypothesis) H0 when a specific [alternative hypothesis](https://en.wikipedia.org/wiki/Alternative_hypothesis) H_1 is true. It is commonly denoted by ![1-\beta ](https://wikimedia.org/api/rest_v1/media/math/render/svg/50cbf597b2f73a22464f1fee6d541574cb044c55)





![007.jpg](https://github.com/dongzhang84/Study_Notes/blob/main/figures/AB_testing/007.jpg?raw=true)

![n equation](https://ceblog.s3.amazonaws.com/wp-content/uploads/2016/05/03115330/n-equation.png)

#### Common test statistics

![008.png](https://github.com/dongzhang84/Study_Notes/blob/main/figures/AB_testing/008.png?raw=true)



#### Metrics

![009.jpg](https://github.com/dongzhang84/Study_Notes/blob/main/figures/AB_testing/009.jpg?raw=true)

Examples:

- Def #1 (Cookie probability): For each time interval, number of cookies that click divided by number of cookies

- Def #2 (Pageview probability): Number of pageviews with a click within time interval divided by number of pageviews

- Def #3 (Rate): Number of clicks divided by number of pageviews

![010.jpg](https://github.com/dongzhang84/Study_Notes/blob/main/figures/AB_testing/010.jpg?raw=true)

![011.jpg](https://github.com/dongzhang84/Study_Notes/blob/main/figures/AB_testing/011.jpg?raw=true)





## 8. Bayesian Inference

Bayesian inference is a method of statistical inference in which Bayes' theorem is used to update the probability for a hypothesis as more evidence or information becomes available.



### 8.1) Baye's Theorem

**Bayesian Probability**

![{\displaystyle P(A\mid B)={\frac {P(A\cap B)}{P(B)}},{\text{ if }}P(B)\neq 0,}](https://wikimedia.org/api/rest_v1/media/math/render/svg/7b424233f9f41ed1e6e96deecab00e0e158029ec)

So we have

![{\displaystyle P(A\mid B)={\frac {P(B\mid A)P(A)}{P(B)}}}](https://wikimedia.org/api/rest_v1/media/math/render/svg/87c061fe1c7430a5201eef3fa50f9d00eac78810)

For continuous random variables: 

![{\displaystyle f_{X\mid Y=y}(x)={\frac {f_{Y\mid X=x}(y)f_{X}(x)}{f_{Y}(y)}}.}](https://wikimedia.org/api/rest_v1/media/math/render/svg/e94271d4d6e4727af54969fbedecccd5456b34e0)

![img](https://upload.wikimedia.org/wikipedia/commons/thumb/7/74/Bayes_theorem_simple_example_tree.svg/2560px-Bayes_theorem_simple_example_tree.svg.png)



### Prior and Posterior Distribution

The posterior probability is the probability of the parameters ![\theta ](https://wikimedia.org/api/rest_v1/media/math/render/svg/6e5ab2664b422d53eb0c7df3b87e1360d75ad9af) given the evidence ![X](https://wikimedia.org/api/rest_v1/media/math/render/svg/68baa052181f707c662844a465bfeeb135e82bab): ![{\displaystyle p(\theta |X)}](https://wikimedia.org/api/rest_v1/media/math/render/svg/2594603c1c2b622471d9a19d1ea54daa152026b4).

It contrasts with the [likelihood function](https://en.wikipedia.org/wiki/Likelihood_function), which is the probability of the evidence given the parameters: ![p(X|\theta )](https://wikimedia.org/api/rest_v1/media/math/render/svg/0f1665d485e91e5a3e953a937574c78389668777).



