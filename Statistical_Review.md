# Statistical Review

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

## 2. Hypothesis Tests

A statistical hypothesis is a hypothesis that is testable on the basis of observed data modeled as the realized values taken by a collection of random variables. 

### 2.1) Steps

(From David Spiegelhalter's "The Art of Statistics")

1. Set up a question in terms of a **null hypothesis** that we want to check. This is generally given the notation H0.
2. Choose a **test statistic** that estimates something  that, if it turned out to be extreme enough, would lead us to double the null hypothesis. 
3. Generate the sampling distribution of this test statistic, were the **null hypothesis** true. 
4. Check whether our observed statistic lies in the tails of this distribution and summarize this by the **P-value**: the probability, were the null hypothesis true, of observing such an extreme statistic. The **P-value** is therefore a particular tail-area.
5. Declare the result statistically significant if the **P-value** is below some critical threshold.  



### 2.2) Test Statistic Distribution

In standard cases this will be a well-known result. For example,

- [student's t-distribution](https://en.wikipedia.org/wiki/Student%27s_t-distribution) (known degrees of freedom)
- [normal distribution](https://en.wikipedia.org/wiki/Normal_distribution) (known mean and variance)



### 2.3) Various Tests

----

#### 2.3.1. Z-test

A Z-test is any statistical test for which the distribution of the test statistic under the null hypothesis can be approximated by a **normal distribution**.



1.a **One Sample Z-test**

1.b **Two Sample Z-test**

Two normally distributed but independent populations, $\sigma$ is known. 





------

#### 2.3.2. T-test

The t-test is any statistical hypothesis test in which the test statistic follows a **Student's t-distribution** under the null hypothesis.



2.a **One Sample and Two Sample T-tests**:

- A **one-sample**  location test of whether the mean of a population has a value specified in a null hypothesis.

- A **two-sample** location test of the **null hypothesis** such that **the means of two populations are equal**.

  

2.b **Unpaired and Paired two-sample T-tests** 

Two-sample *t*-tests for a difference in mean involve **independent samples (unpaired samples)** or paired samples.

2.c **Correlated (or Paired) T-Test**

2.d **Equal Variance (or Pooled) T-Test**

2.e **Unequal Variance T-Test**



Demonstration

![T-test](https://www.investopedia.com/thmb/J-NC9PcSiu_SrtLk59nYeJqHQFA=/6250x4623/filters:no_upscale():max_bytes(150000):strip_icc():format(webp)/ttest22-0afd4aefe9cc42628f603dc2c7c5f69a.png)



----

#### 2.3.3. Chi-Squared Tests

3.a **Chi-Squared Test of Association/Goodness-of-Fit Test** 

Non-normal, and non-t distribution. But an assumed model distribution. 

A statistical test that indicates the degree of incompatibility of data with an assumed statistical model comprising the null hypothesis, which may be one of lack of association, or some other specified mathematical form. Specifically, the test compares a set of $m$ Observed counts $o_1, o_2, o_3, o_4 ... o_m$ With a set of expected values $e_1, e_2, e_3, ... e_m$, which have been calculated under the null hypothesis. The simplest version of the test statistic is given as
$$
\begin{equation}
X^2 = \sum_{j=1}^{m} \frac{(o_j - e_j)^2}{e_j}
\end{equation}
$$
Under the null hypothesis $X^2$ Will have an approximate chi-square sampling distribution, enabling an associated  **P-value** to be calculated. 



3.b **Chi-Squared Test of Variance**

A chi-square test ( [Snedecor and Cochran, 1983](https://www.itl.nist.gov/div898/handbook/eda/section4/eda43.htm#Snedecor)) can be used to test if the variance of a population is equal to a specified value. This test can be either a two-sided test or a one-sided test. 

[Reference](https://www.itl.nist.gov/div898/handbook/eda/section3/eda358.htm)

----

#### 2.3.4. Regression T-test

Some references to read

https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLSResults.t_test.html

https://stackoverflow.com/questions/27928275/find-p-value-significance-in-scikit-learn-linearregression



-----

#### 2.4) Errors and Estimate

**P-value**

In null hypothesis significance testing, the p-value is the probability of obtaining test results at least as extreme as the results actually observed, under the assumption that the null hypothesis is correct.



**Two errors**

- **Type I error**: a type I error is the mistaken rejection of the null hypothesis (probability: $\alpha$).
- **Type II error**: a type II error is the mistaken acceptance of the null hypothesis (probability: $\beta$).

**Significant level**: $\alpha$

**Statistical power**: 1- $\beta$





## 3. ANOVA (Analysis of Variance)

### One-way ANOVA

The one-way analysis of variance (ANOVA) is used to determine whether there are any statistically significant differences between the means of three or more independent (unrelated) groups. 

![One-way ANOVA Null Hypothesis](https://statistics.laerd.com/statistical-guides/img/one-way-anova-null-hypothesis.png)

See example here:

https://en.wikipedia.org/wiki/One-way_analysis_of_variance



### Two-way ANOVA

A two-way ANOVA is used to estimate how the mean of a quantitative variable changes according to the levels of two different categorical independent variables. 



## 4. Central Limited Theorem

If $X_1, X_2, X_3, ..., X_n$ are $n$ random samples drawn from a population with overall mean $\mu$ And finite variance $\sigma^2$, and if $\bar{X}_n$ Is the sample mean, then the limiting form of the distribution,

 ![{\textstyle Z=\lim _{n\to \infty }{\sqrt {n}}{\left({\frac {{\bar {X}}_{n}-\mu }{\sigma }}\right)}}](https://wikimedia.org/api/rest_v1/media/math/render/svg/d269e67503670688dba6d8c25481f5e63c0b1d5b), 

is a standard normal distribution.



## 5. A/B Testing

A/B testing (also known as bucket testing or split-run testing) is a user experience research methodology. 

A/B tests consist of a randomized experiment with two variants, A and B. It includes application of statistical hypothesis testing or "two-sample hypothesis testing" as used in the field of statistics. A/B testing is a way to compare two versions of a single variable, typically by testing a subject's response to variant A against variant B, and determining which of the two variants is more effective.

![img](https://upload.wikimedia.org/wikipedia/commons/2/2e/A-B_testing_example.png)



## Bayesian Inference

Bayesian inference is a method of statistical inference in which Bayes' theorem is used to update the probability for a hypothesis as more evidence or information becomes available.
