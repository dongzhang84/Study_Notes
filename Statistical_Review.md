# Statistical Review

1. Probability Distributions

   - Discrete Distributions

     - Binomial Distribution, Bernoulli Distribution, Poisson Distribution

   - Continuous Distributions

     - Uniform Distribution, Exponential Distribution, Normal Distribution

   - Multivariate Normal

     

2. Sampling and Basic EDA

   - Some Definitions
     - Mean, Median, Standard Deviation, Standard Deviation, Standard Error
   - Basic EDA
     - Q-Q Plot

   

3. Hypothesis Testing 

   - Steps, Errors and Estimate (p-value), Two Errors, Significant Level and Statistical Power, Test Statistic Distributions
   - **Z-Test**: Confidence Interval, One-sample and Two-sample Z-tests, Pooled and Un-pooled two-sample Z-tests
   - **T-Test**: One-sample and two-sample T-tests, Pooled and Un-pooled T-tests
   - **Chi-Squared Tests**: Chi-squared test of association/Goodness-of-fit, test for categorical data, Test of variance

   

4. Central Limit Theorem
   

5. Linear Regression

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
   - 



## 1. Probability Distributions

1.1 **Discrete Distributions**

- Binomial Distribution

  ![{\displaystyle f(k,n,p)=\Pr(k;n,p)=\Pr(X=k)={\binom {n}{k}}p^{k}(1-p)^{n-k}}](https://wikimedia.org/api/rest_v1/media/math/render/svg/b872c2c7bfaa26b16e8a82beaf72061b48daaf8e)

  ![Probability mass function for the binomial distribution](https://upload.wikimedia.org/wikipedia/commons/thumb/7/75/Binomial_distribution_pmf.svg/300px-Binomial_distribution_pmf.svg.png)

- Bernoulli Distribution

  ![{\displaystyle f(k;p)={\begin{cases}p&{\text{if }}k=1,\\q=1-p&{\text{if }}k=0.\end{cases}}}](https://wikimedia.org/api/rest_v1/media/math/render/svg/5fff3412509e73816dc2b28405b93c34f89ee487)

  

- Poisson Distribution

  ![\!f(k; \lambda)= \Pr(X{=}k)= \frac{\lambda^k e^{-\lambda}}{k!},](https://wikimedia.org/api/rest_v1/media/math/render/svg/6c429d187b5d4ef8ddea32a2d224f423cf9fe5b0)

  ![Poisson pmf.svg](https://upload.wikimedia.org/wikipedia/commons/thumb/1/16/Poisson_pmf.svg/325px-Poisson_pmf.svg.png)



1.2 **Continuous Distributions**

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



## 2. Sampling

### 2.1 Some Definitions

**Sample Mean**

![mean.png](https://github.com/dongzhang84/Study_Notes/blob/main/figures/statistics/mean.png?raw=true)

**Standard Deviation**

Discrete: 

![{\displaystyle \sigma ={\sqrt {{\frac {1}{N}}\sum _{i=1}^{N}(x_{i}-\mu )^{2}}},{\text{ where }}\mu ={\frac {1}{N}}\sum _{i=1}^{N}x_{i}.}](https://wikimedia.org/api/rest_v1/media/math/render/svg/98f02417b7c2830d941364f6b40e22ea63a9dd1f)

Continuous: 

![{\displaystyle {\begin{aligned}\sigma &\equiv {\sqrt {\operatorname {E} \left[(X-\mu )^{2}\right]}}={\sqrt {\int _{-\infty }^{+\infty }(x-\mu )^{2}f(x)dx}},\end{aligned}}}](https://wikimedia.org/api/rest_v1/media/math/render/svg/50fc45dec3e6e45e6b06bb50ea4d218269049d94)



**Standard Error**

![{\displaystyle {\sigma }_{\bar {x}}\ ={\frac {\sigma }{\sqrt {n}}}}](https://wikimedia.org/api/rest_v1/media/math/render/svg/f9dac77577c2717cbb973388e4d6563915705742)

![SE.png](https://github.com/dongzhang84/Study_Notes/blob/main/figures/statistics/SE.png?raw=true)



### 2.2 Basic EDA

#### 2.2.1 Q-Q Plot

A Q–Q (quantile-quantile) plot is a probability plot, which is a graphical method for comparing two probability distributions by plotting their quantiles against each other.

For example:

![img](https://upload.wikimedia.org/wikipedia/commons/thumb/0/08/Normal_normal_qq.svg/300px-Normal_normal_qq.svg.png)





## 3. Hypothesis Tests

A statistical hypothesis is a hypothesis that is testable on the basis of observed data modeled as the realized values taken by a collection of random variables. 

### 3.1) Steps

(From David Spiegelhalter's "The Art of Statistics")

1. Set up a question in terms of a **null hypothesis** that we want to check. This is generally given the notation H0.
2. Choose a **test statistic** that estimates something  that, if it turned out to be extreme enough, would lead us to double the null hypothesis. 
3. Generate the sampling distribution of this test statistic, were the **null hypothesis** true. 
4. Check whether our observed statistic lies in the tails of this distribution and summarize this by the **P-value**: the probability, were the null hypothesis true, of observing such an extreme statistic. The **P-value** is therefore a particular tail-area.
5. Declare the result statistically significant if the **P-value** is below some critical threshold.  





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



##### 1.a) **Confidence Interval**

![CI = \bar{x} \pm z \frac{s}{\sqrt{n}}](https://www.gstatic.com/education/formulas2/397133473/en/confidence_interval_formula.svg) 

- | C    | z*    |
  | ---- | ----- |
  | 99%  | 2.576 |
  | 98%  | 2.326 |
  | 95%  | 1.96  |
  | 90%  | 1.645 |

![img](https://upload.wikimedia.org/wikipedia/commons/b/bb/Normal_distribution_and_scales.gif)



1.b **One Sample Z-test**

![z_test1.png](https://github.com/dongzhang84/Study_Notes/blob/main/figures/statistics/z_test1.png?raw=true)

- **Step 1**: State the Null Hypothesis. 

- **Step 2**: Use the z-formula to find a z-score.
  ![One Sample Z Test](https://www.statisticshowto.com/wp-content/uploads/2014/02/z-score-formula.jpg)

- **Step 3**: Compute P-value.
  For example, Tabled value for *z* ≤ 1.28 is 0.8997

  P-value is 1 – 0.8997 = 0.1003

- **Step 4**: Evaluate null hypothesis. 

![Two Sample Z Test of Proportions](https://sixsigmastudyguide.com/wp-content/uploads/2021/01/zp7-1.png)



##### 1.c) **Two Sample Z-test** [reference](https://sixsigmastudyguide.com/two-sample-test-of-proportions/)

**Requirements**: Two normally distributed but independent populations, σ is known.

![Two Sample Z Test of Proportions](https://sixsigmastudyguide.com/wp-content/uploads/2021/01/zp1.png)

**Binomial Example**

- **Pooled Z test of proportions formula**

![img](https://sixsigmastudyguide.com/wp-content/uploads/2021/01/zp2-1.png)

where ![The one and two sample proportion hypothesis ](https://sixsigmastudyguide.com/wp-content/uploads/2019/10/on7.png)

- **Un-pooled Z test of proportions formula**

![Two Sample Z Test of Proportions](https://sixsigmastudyguide.com/wp-content/uploads/2021/01/zp3-1.png)

**Z** – ![img](https://www.analystsoft.com/en/products/statplus/content/help/analysis_basic_statistics_two_sample_z-test_files/image004.png) critical value for z.

**P(Z<=z)  p-level** – probability of observing the sample statistic as extreme as the test statistic. 



**Unpair Z-test**

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

Non-normal, and non-t distribution. But an assumed model distribution. 

A statistical test that indicates the degree of incompatibility of data with an assumed statistical model comprising the null hypothesis, which may be one of lack of association, or some other specified mathematical form. Specifically, the test compares a set of $m$ Observed counts $o_1, o_2, o_3, o_4 ... o_m$ With a set of expected values $e_1, e_2, e_3, ... e_m$, which have been calculated under the null hypothesis. The simplest version of the test statistic is given as
$$
\begin{equation}
X^2 = \sum_{j=1}^{m} \frac{(o_j - e_j)^2}{e_j}
\end{equation}
$$
Under the null hypothesis $X^2$ Will have an approximate chi-square sampling distribution, enabling an associated  **P-value** to be calculated. 



**Example: Compare Number of Real data to Expected Numbers**

![chi_squred_2.png](https://github.com/dongzhang84/Study_Notes/blob/main/figures/statistics/chi_squred_2.png?raw=true)

![img](https://www.jmp.com/en_ch/statistics-knowledge-portal/chi-square-test/chi-square-goodness-of-fit-test/_jcr_content/par/styledcontainer_2069/par/image_523413870.img.png/1623884285244.png)





**Example chi-squared test for categorical data**

![chi_squred_3.png](https://github.com/dongzhang84/Study_Notes/blob/main/figures/statistics/chi_squred_3.png?raw=true)



3.b **Chi-Squared Test of Variance**

A chi-square test ( [Snedecor and Cochran, 1983](https://www.itl.nist.gov/div898/handbook/eda/section4/eda43.htm#Snedecor)) can be used to test if the variance of a population is equal to a specified value. This test can be either a two-sided test or a one-sided test. 

[Reference](https://www.itl.nist.gov/div898/handbook/eda/section3/eda358.htm)

![chi_squred_4.png](https://github.com/dongzhang84/Study_Notes/blob/main/figures/statistics/chi_squred_4.png?raw=true)

![chi_squred_5.png](https://github.com/dongzhang84/Study_Notes/blob/main/figures/statistics/chi_squred_5.png?raw=true)





## 4. Central Limited Theorem

If $X_1, X_2, X_3, ..., X_n$ are $n$ random samples drawn from a population with overall mean $\mu$ And finite variance $\sigma^2$, and if $\bar{X}_n$ Is the sample mean, then the limiting form of the distribution,

 ![{\textstyle Z=\lim _{n\to \infty }{\sqrt {n}}{\left({\frac {{\bar {X}}_{n}-\mu }{\sigma }}\right)}}](https://wikimedia.org/api/rest_v1/media/math/render/svg/d269e67503670688dba6d8c25481f5e63c0b1d5b), 

is a standard normal distribution.

(Need to expand more)



## 5. Linear Regression

### 5.1) Assumptions of Linear Regression

Linear regression is an analysis that assesses whether one or more predictor variables explain the dependent (criterion) variable.  The regression has five key assumptions:

- Linear relationship
- Multivariate normality
- No or little multicollinearity
- No auto-correlation
- Homoscedasticity



**Linear relationship:** There exists a linear relationship between the independent variable, x, and the 

**Independence:** The residuals are independent. In particular, there is no correlation between consecutive residuals in time series data.

**Normality**：The residuals of the model are normally distributed.

**Homoscedasticity:** The residuals have constant variance at every level of x.

For example: 

![img](https://upload.wikimedia.org/wikipedia/commons/9/93/Homoscedasticity.png)

Plot with random data showing homoscedasticity: at each value of x, the y-value of the dots has about the same variance.



### 5.2) Simple Linear Regression



Consider a model function: 

![y=\alpha +\beta x,](https://wikimedia.org/api/rest_v1/media/math/render/svg/bf2c1cac7c1e6c9a426d92e9adad6ff4d8b4152e)

We can describe the underlying relationship between $y_i$ and $x_i$  as

![ y_i = \alpha + \beta x_i + \varepsilon_i.](https://wikimedia.org/api/rest_v1/media/math/render/svg/968be557dd22b1a2e536b8d22369cfdb37f58703)

And the solution of the parameters are

![{\textstyle {\begin{aligned}{\widehat {\alpha }}&={\bar {y}}-({\widehat {\beta }}\,{\bar {x}}),\\[5pt]{\widehat {\beta }}&={\frac {\sum _{i=1}^{n}(x_{i}-{\bar {x}})(y_{i}-{\bar {y}})}{\sum _{i=1}^{n}(x_{i}-{\bar {x}})^{2}}}\\[6pt]&={\frac {s_{x,y}}{s_{x}^{2}}}\\[5pt]&=r_{xy}{\frac {s_{y}}{s_{x}}}.\\[6pt]\end{aligned}}}](https://wikimedia.org/api/rest_v1/media/math/render/svg/9caed0f59417a425c988764032e5892130e97fa4)

Note that 

![{\displaystyle {\begin{aligned}{\widehat {\beta }}&={\frac {\sum _{i=1}^{n}(x_{i}-{\bar {x}})(y_{i}-{\bar {y}})}{\sum _{i=1}^{n}(x_{i}-{\bar {x}})^{2}}}={\frac {\sum _{i=1}^{n}(x_{i}-{\bar {x}})^{2}*{\frac {(y_{i}-{\bar {y}})}{(x_{i}-{\bar {x}})}}}{\sum _{i=1}^{n}(x_{i}-{\bar {x}})^{2}}}\\[6pt]\end{aligned}}}](https://wikimedia.org/api/rest_v1/media/math/render/svg/e1ac1b7ef40d1c91a192327f20ae7ca88f4c4d37)



### 5.3) Metrics of model evaluation

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







### 5.4) Regression test

**T-Test**

![regress_t.png](https://github.com/dongzhang84/Study_Notes/blob/main/figures/statistics/regress_t.png?raw=true)

**F-Test**

![regress_f.png](https://github.com/dongzhang84/Study_Notes/blob/main/figures/statistics/regress_f.png?raw=true)

Some references to read

https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLSResults.t_test.html

https://stackoverflow.com/questions/27928275/find-p-value-significance-in-scikit-learn-linearregression



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

where

SSR = the regression sum of squares

SSE = the error sum of squares

SST = the total sum of squares (SST = SSR + SSE)

dfr = the model degrees of freedom (equal to dfr = *k* - 1)

dfe = the error degrees of freedom (equal to dfe = *n* - *k* - 1)

*k* = the total number of groups (levels of the independent variable)

*n* = the total number of valid observations

dfT = the total degrees of freedom (equal to dfT = dfr + dfe = *n* - 1)

MSR = SSR/dfr = the regression mean square

MSE = SSE/dfe = the mean square error

Then the F statistic itself is computed as

F=MSR/MSE

See example here:

https://en.wikipedia.org/wiki/One-way_analysis_of_variance



### 6.2) Two-way ANOVA

A two-way ANOVA is used to estimate how the mean of a quantitative variable changes according to the levels of two different categorical independent variables. 

A two-way ANOVA with interaction tests three null hypotheses at the same time:

- There is no difference in group means at any level of the first independent variable.
- There is no difference in group means at any level of the second independent variable.
- The effect of one independent variable does not depend on the effect of the other independent variable (a.k.a. no interaction effect).

https://www.itl.nist.gov/div898/handbook/prc/section4/prc437.htm

![anova_1.png](https://github.com/dongzhang84/Study_Notes/blob/main/figures/statistics/anova_1.png?raw=true)

![anova_2.png](https://github.com/dongzhang84/Study_Notes/blob/main/figures/statistics/anova_2.png?raw=true)



See this example: https://www.statology.org/two-way-anova/



## 7. A/B Testing

A/B testing (also known as bucket testing or split-run testing) is a user experience research methodology. 

A/B tests consist of a randomized experiment with two variants, A and B. It includes application of statistical hypothesis testing or "two-sample hypothesis testing" as used in the field of statistics. A/B testing is a way to compare two versions of a single variable, typically by testing a subject's response to variant A against variant B, and determining which of the two variants is more effective.

![img](https://upload.wikimedia.org/wikipedia/commons/2/2e/A-B_testing_example.png)



### Types of Metrics

- Click through rate = # clicks / # visits

- Click through probability = # unique visitors who click / # unique visitors

- Retention rate = # users remained at the end of evaluation period / # users at the beginning of the evaluation period

- Churn rate = # users acquired / # users lost

- ROI = (current value – investment) / investment





**Marginal Error**

Marginal error (M) = Z score of the confidence interval * Standard Error (SE)

Marginal error is function of proportion of success and sample size – means need to consider the proportion of success when deciding the sample size.

![001.jpg](https://github.com/dongzhang84/Study_Notes/blob/main/figures/AB_testing/001.jpg?raw=true)

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

- **Significant level**: alpha

- **Statistical Power**: beta

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

### 8.1 Baye's Theorem

**Bayesian Probability**

![{\displaystyle P(A\mid B)={\frac {P(A\cap B)}{P(B)}},{\text{ if }}P(B)\neq 0,}](https://wikimedia.org/api/rest_v1/media/math/render/svg/7b424233f9f41ed1e6e96deecab00e0e158029ec)

So we have

![{\displaystyle P(A\mid B)={\frac {P(B\mid A)P(A)}{P(B)}}}](https://wikimedia.org/api/rest_v1/media/math/render/svg/87c061fe1c7430a5201eef3fa50f9d00eac78810)

For continuous random variables: 

![{\displaystyle f_{X\mid Y=y}(x)={\frac {f_{Y\mid X=x}(y)f_{X}(x)}{f_{Y}(y)}}.}](https://wikimedia.org/api/rest_v1/media/math/render/svg/e94271d4d6e4727af54969fbedecccd5456b34e0)

![img](https://upload.wikimedia.org/wikipedia/commons/thumb/7/74/Bayes_theorem_simple_example_tree.svg/2560px-Bayes_theorem_simple_example_tree.svg.png)

