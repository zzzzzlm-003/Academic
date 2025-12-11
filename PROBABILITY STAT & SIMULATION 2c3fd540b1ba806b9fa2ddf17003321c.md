# PROBABILITY STAT & SIMULATION

Concepts & Code

IEORE4101

# 6. Simulation & Analysis

## 6.1 Generate samples from a distribution

### Inverse Transform Method

<aside>
📏

 The primary goal is to find the inverse function of the CDF, $F^{-1}(u)$ , to transform a uniform random variable $U \sim Unif(0, 1)$ into a random variable $X = F^{-1}(U)$ from the desired distribution.

</aside>

- **Continuous Distribution**
    1. def PDF
    2. def CDF
        - Closed-Form Solution
            
            `cdf = sp.integrate(pdf,(x, lower, upper))`
            
        - Analytical Solution
            
            ```python
            def cdf_func(x, lm):
            	x = np.linspace(0,x,1000)
            	pdf = lm* np.exp(-lm*x)
            	return **np.trapezoid(pdf, x)**
            ```
            
    3. Solve for the Inverse CDF
        - Closed-Form Solution
            
            `sp.solve(cdf-u , x)[0]`
            
            - If [] is returned from sp.solve(). There is no closed-form solution for the inverse cdf
                - We use pynverse package to perform numeric inversion 👇
        - Brent's Method
            
            `pn.inversefunc(func_cdf, y_values, (lower,upper), args)`
            
            - `domain=(lower, upper)`: Defines the range within which the search for the inverse should be performed. For the inverse transform method, this corresponds to the range of *x*. If there is no lower or upper bound, simply use None.
            - `args :` The args parameter allows you to pass extra arguments that are required by your function. For example, if your function is f(x, a, b), and you want to fix a=2 and b=3, then you should pass args=(2, 3).
    
- **Discrete Distribution**
    1. Compare each cdf with u using a loop
    2. Stop the loop when we find a cdf that is higher than u
    3. Return the corresponding x
    
    ```python
    def inverse_sampling(lm):
        u = rng.random() 
        x = 0 
        while True:
            if u < cdf(x): 
            ## if find closest cdf u should round up to return the corresponding x 
                return x 
            else: # otherwise, find the next x and try again 
                x = x+1 
    ```
    
    Example
    
    **Setting**
    
    <aside>
    📏
    
    > Assume that the number of units $y$ a consumer purchases, given their $z$ value, is randomly sampled from $Y$ with the following **unnormalized** PMF:
    > 
    
    $P(y \mid z) \propto \left(\frac{1}{y + 1}\right)^z,   y = 0, 1, 2, \dots, 10.$
    
    > In addition, $z$ for a customer is randomly sampled from $Z$ with the following **normalized** PDF:
    > 
    
    $f(z) = \frac{\exp(1 - z)}{1 - \exp(-2)},   1 < z < 3.$
    
    Construct a function with argument $z$ to sample from $Y$ **given a *z* value. Inside the function, use the inverse transform method to generate a sample of *Y* 
    
    </aside>
    
    ```python
    def sampling_y(z):
        y = np.arange(11)
        pmf = 1/(y+1)**z 
        pmf  = pmf/np.sum(pmf) #sum = 1
        cdf = np.cumsum(pmf)
        u = rng.random()
        sample = 0
        while True:
            if u<cdf[sample]:
                break 
            else:
                sample=sample+1 
        return sample 
        
    y_samples = [sampling_y(1.1) for i in range(10000)]
    y_values, y_freq = np.unique(y_samples, return_counts=True)
    plt.bar(y_values, y_freq/np.sum(y_freq))
    ```
    
    **common cdf functions**
    
    > `stats.poisson.cdf(lm)`
    `stats.bernoulli.cdf(k=1, p=0.6)`
    `stats.binom.cdf(k=3, n=10, p=0.5)`
    `stats.geom.cdf(k=5, p=0.2)`
    `stats.nbinom.cdf(k=3, n=5, p=0.5)` #n success, k  failure
    `stats.randint.cdf(k=4, low=1, high=7)`
    > 
    
- **Bi-variate distribution**
    
    $f_{X,Y}(x, y) = f_X(x) \cdot f_{Y|X}(y|x)$
    
1. Sample x from Marginal distribution (X)
2. Sample y from conditional distribution (Y|x)
3. Derive marginal and conditional from joint
    
    ```python
    import numpy as np
    
    # assume 1. X ~ Exponential(lambda_x=2)  # marginal
    # 2. Y|x ~ Exponential(lambda_y|x = 3 * x) # conditional
    
    rng = np.random.default_rng(42)
    N_samples = 1000
    
    # 1. Sample X from Marginal dist (X)
    lambda_x = 2
    scale_x = 1 / lambda_x
    samples_X = rng.exponential(scale=scale_x, size=N_samples) # 抽取 X 的样本
    
    samples_Y = []
    for x_val in samples_X:
        # 2. Sample Y from conditional dist (Y|x)
        # scale_y|x = 1 / lambda_y|x = 1 / (3 * x_val)
        scale_y_given_x = 1 / (3 * x_val) 
        
        y_val = rng.exponential(scale=scale_y_given_x, size=1)[0] 
        
        samples_Y.append(y_val)
    ```
    
- Rng.distribution name to generate samples from common distributions directly
    
    ```python
    rng = np.random.default_rng(20) 
    #define a random number generator ,seed = 20
    u = rng.random(10) 
    ~~Verify correctness~~
    ```
    
    <aside>
    📏
    
    1. **`rng.poisson(lam=1.0, size=None)`**
    2. **`rng.choice(a, size=None, p=None)`** 
        
        ## p by default assumes uniform probability
        
    3. **`rng.geometric(p, size=None)`** 
        
        ##number of trials needed to see the first success
        
    4. **`rng.binomial(n, p, size=None)`**
    5. **`rng.negative_binomial(n, p, size=None)`** 
        
        ## number of failures needed to see *n* successes
        
    6. **`rng.uniform(low=0.0, high=1.0, size=None)`**
    7. **`rng.exponential(scale=**1/*λ***, size=None)**` 
    8. **`rng.normal(loc=0.0, scale=1.0, size=None)`**
    9. **`rng.beta(a, b, size=None)`**
    10. **`rng.gamma(shape, scale, size=None)`**
    </aside>
    

- Managerial simulation
    - Construction function A that simulate the system once
    - Construction function B that repeats function A many times to generate many independent samples from the output distribution
        - inference : summary statistics, visualization, hypothesis testing, confidence intervals.
    - Repeat function B under various policies to choose the policy that gives the best output
    - 

### Inference

- Summary statistics
    - mean, variance, standard deviation, mode, median, percentiles, correlation, covariance,...
    - Compute sample statistics and compare them to the true value
    
    ```python
    #set up
    lm = 2
    rng = np.random.default_rng(10)
    samples_exp = rng.exponential(1/lm,100)
    
    #mean
    print(np.mean(samples_exp))
    print(1/lm) #population
    
    #mode 
    #only discrete dist meaningful
    sample_poisson = rng.poisson(5,100)
    print(stats.mode(sample_poisson))
    
    #median
    print(np.median(sample_poisson))
    
    #variance
    print(np.var(samples_exp, ddof=1)) 
    ### ddof
    print(1/lm**2) #population
    
    #standard deviation
    print(np.std(samples_exp, ddof=1))
    
    #range
    print(np.max(samples_exp)-np.min(samples_exp))
    
    #percentile
    np.percentile(samples_exp,[25,50,75])
    true_percentile = stats.poisson.ppf(0.25, mu=TRUE_LAMBDA)
    
    # covariance (ddof=1)
    sample_cov_matrix = np.cov(input_demand, output_profit, ddof=1)
    sample_covariance = sample_cov_matrix[0, 1]
    
    # correlation
    sample_corr_matrix = np.corrcoef(input_demand, output_profit)
    sample_correlation = sample_corr_matrix[0, 1]
    ```
    

### Visualization

<aside>
📏

Visualize the sample distribution and population (true) in the same graph

</aside>

- Empirical pdf (continuous)
    
     `plt.**hist**(x, density=True, bins =)`
    
- Empirical pmf (discrete)
    - `plt.**bar**(x,y)`
        - `unique_x, count_x = np.unique(sample, return_counts=True)
        plt.bar(unique_x, count_x/np.sum(count_x))`
    - `plt.**scatter**(x,y)`
- Empirical cdf
    - `CDF_values = stats.ecdf(samples_exp).cdf 
    x_v = CDF_values.quantiles
    cdf = CDF_values.probabilities
    plt.step(x, cdf, where="post")`

**Example:**

Discrete  (Poisson)

```python
samples_poisson = rng.poisson(lam=5, size=1000)
mu = 5

# 1. sample PMF (true)
unique_x, count_x = np.unique(sample_poisson, return_counts=True)
plt.bar(unique_x, count_x / np.sum(count_x), 
        label='Sample PMF', alpha=0.6, color='skyblue')
#np.sum(count_x) or len(samples)

# 2. population PMF (theoretical) 
mu = np.mean(sample_poisson) # or lambda
x = np.arange(np.min(unique_x), np.max(unique_x) + 1)
theoretical_pmf = stats.poisson.pmf(x, mu=mu)
plt.plot(x_range, theoretical_pmf, 'r--', 
         marker='o', markersize=4, label='Population PMF') 
#plot or scatter     
```

Continuous (Exponential)

```python
samples_exp = rng.exponential(scale=1/lm, size=1000)
lm = 2

# 1. sample PMF (true)
plt.hist(samples_exp, density=True, bins=20, 
         label='Sample PDF', alpha=0.6, color='skyblue')

# 2. population PMF (theoretical) 
# define x (z_array/x)
x = np.linspace(np.min(samples_exp), np.max(samples_exp), 100)
y = stats.expon.pdf(x, scale=1/lm) 

plt.plot(x, y, 'r-', linewidth=2, label='Population PDF')

plt.legend()  
plt.title('Sample vs Population')
plt.show()
```

- Central Limit Theorem: if we have a large number of i.i.d samples, the sample mean X follow approximately a normal distribution (, 2/n), where  and 2 are mean and variance of the original distribution. n is the sample size
    - Backbone of confidence interval and t-test

## 📊 Estimator & Confidence Interval

| **Estimator** | **Target** | **df** | **SE** |  |
| --- | --- | --- | --- | --- |
| $\bar{X}$  | $\mu$ (population) | $n-1$ | $\mathbf{s / \sqrt{n}}$ | One Sample T test, **CI** `stats.t.ppf(1-alpha/2, df` |
| $s$ | $\sigma$ (population) | N/A |  | s doesn’t follow normal distribution |
| $\hat{\beta}_k$  | $\beta_k$ | $\mathbf{n - p - 1}$ | $\mathbf{model.bse}$  |  |
| $g(\hat{\beta})$  | $g(\beta)$ | $\mathbf{n - p - 1}$ | $\mathbf{\sqrt{\text{Var}_{\text{est}}}}$ (Delta Method ) | **Delta Method T test** |
| $\bar{X}_1 - \bar{X}_2$ | $\mu_1 - \mu_2$ | **If** $\sigma_1=\sigma_2$ $\rightarrow n_1 + n_2 - 2$ | **If** $\sigma_1=\sigma_2$ $\rightarrow$ **Pooled Variance** | Two Sample T test |
| $\bar{X}_1 - \bar{X}_2$ | $\mu_1 - \mu_2$ | $\mathbf{df}$ (Welch) | **If** $\sigma_1 \neq \sigma_2$ $\rightarrow$ not pool (Welch's Test) | Two Sample T test |

---

### 1. T-Critical Value

$t_{\alpha/2, df} \quad \text{where } \alpha = 1 - \text{Confidence Level}$

`alpha = 1 - confidence_level
t_critical = stats.t.ppf(1 - alpha / 2, df=df) #`

### 2. degree of freedom

Regression: $\mathbf{n - p - 1}$

- $n$:  Rows of Data
- $p$:  Number of independent variable

# 7.Hypothesis Testing

## T-test

### **One sample t-test**    $t = \frac{\bar{x} - \mu_0}{s / \sqrt{n}}$

- One set of samples
- Samples can come from any distribution as long as the sample size is large
- Example
    
    ```python
    import numpy as np
    import scipy.stats as stats
    
    # assumption
    n = 50
    x_bar = 9.9
    s = 0.8
    mu_0 = 10
    
    # ---  (Manual Calculation) ---
    # 1. Standard Error
    se = s / np.sqrt(n)
    
    # 2.  T-score
    t_score = (x_bar - mu_0) / se  
    # 输出: -0.8839
    
    # 3.  P-value 
    #  Ha: mu != 10 
    p_value = 2 * stats.t.cdf(-abs(t_score), df=n-1)
    ```
    
    ```python
    # Scipy Function
    stats.ttest_1samp(data, popmean=10, alternative='two-sided')
    ```
    

### **Two sample t-test**

- Two sets of samples
- Samples can come from any distribution as long as the sample size is large
- `scipy.stats.ttest_ind(a, b, equal_var=True, alternative=‘two-sided’)`
- **Unequal Variance**
    - $t = \frac{(\bar{X}_1 - \bar{X}_2) - (\mu_1 - \mu_2)}{\sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}}$
    
    ```python
    import numpy as np
    import math
    
    def calculate_welch_t_statistic(X_bar_1, X_bar_2, mu_1, mu_2, s_sq_1, s_sq_2, n_1, n_2):
        numerator = (X_bar_1 - X_bar_2) - (mu_1 - mu_2)
        variance_term = (s_sq_1 / n_1) + (s_sq_2 / n_2)
        denominator = math.sqrt(variance_term)
        t_statistic = numerator / denominator
        
        return t_statistic
    ```
    
    $$
    df = \frac{\left(\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}\right)^2}{\frac{\left(\frac{s_1^2}{n_1}\right)^2}{n_1 - 1} + \frac{\left(\frac{s_2^2}{n_2}\right)^2}{n_2 - 1}}
    $$
    
    ```python
    def calculate_welch_degrees_of_freedom(s_sq_1, s_sq_2, n_1, n_2):
        # W₁ = s₁²/n₁
        term_1 = s_sq_1 / n_1
        # W₂ = s₂²/n₂
        term_2 = s_sq_2 / n_2
        # (W₁ + W₂)²
        df_numerator = (term_1 + term_2)**2
        
        # Denominator
        # [ (W₁)² / (n₁ - 1) ] + [ (W₂)² / (n₂ - 1) ]
        if n_1 <= 1 or n_2 <= 1:
            raise ValueError
            
        df_denominator = (term_1**2 / (n_1 - 1)) + (term_2**2 / (n_2 - 1))
        df = df_numerator / df_denominator
        return df
    ```
    

- **Equal variance**
    
    $$
    \frac{\bar X_1 - \bar X_2 - (\mu_1 - \mu_2)}
    {\sqrt{s_{\text{pooled}}^{\,2}\left(\frac{1}{n_1} + \frac{1}{n_2}\right)}}
    \;\sim\;
    t_{\,df = n_1 + n_2 - 2}
    $$
    
    $$
    s_{\text{pooled}}^{\,2}
    = \frac{(n_1 - 1)s_1^{\,2} + (n_2 - 1)s_2^{\,2}}
    {\,n_1 + n_2 - 2\,}.
    $$
    
    ```python
    s2_pooled = ((n1-1)*s1**2 + (n2-1)*s2**2) / (n1+n2-2)
    std = np.sqrt(s2_pooled * (1/n1 + 1/n2))
    diff = x1_bar - x2_bar  
    t_score = diff / std
    stats.t.cdf(t_score, df = n1 + n2 - 2)
    ```
    
    Example:
    
    ```python
    data_coupon = AB_Test[AB_Test.test_coupon==1]
    data_nocoupon = AB_Test[AB_Test.test_coupon==0]
    #assuming equal variance 
    stats.ttest_ind(data_coupon.revenue_after, data_nocoupon.revenue_after)
    #assuming unequal variance 
    stats.ttest_ind(data_coupon.revenue_after, \
    data_nocoupon.revenue_after, equal_var=False)
    ```
    

### **T-test for g(beta) in regression**

$H_0: \beta_1 + \beta_2 + \beta_3 + \beta_4 = 0$

```python
from statsmodels.formula.api import ols

model = ols("revenue ~ groupon0 + groupon1 + groupon2 + groupon3 + ...", data).fit()

# def g (Sum of betas)
# g_dev ，[Intercept, b1, b2, b3, b4, ...]
# assumption
g_dev = np.array([0, 1, 1, 1, 1, 0, 0, 0])

# Var(g) = g_dev * Cov_Matrix * g_dev.T
beta_cov = model.cov_params()
g_var = g_dev @ beta_cov @ g_dev.T
g_se = np.sqrt(g_var) 

# 4.  T-score
g_hat = np.sum(model.params.iloc[1:5])
g_true = 0
t_score = (g_hat - g_true) / g_se

# 5. P-value 
df = model.df_resid  #  n - p - 1
p_value = 2 * stats.t.cdf(-abs(t_score), df=df)
```

### **Process**

- **Form Hypothesis**
    - Null Hypothesis ($H_0$): *μ*=5
    - Alternative Hypothesis ($H_a$): *μ*>5
- **Compute the t-Statistic**
    - T_score = estimator - true s.e. (estimator)
        
           $t = \frac{\bar{x} - \mu}{s / \sqrt{n}}$
        
        `t_score = (x_bar - mu) / (s / np.sqrt(n))`
        
- **Compute the P-Value**
    - less: $H_a: \mu_1<\mu_2$  `p_value = 1 - stats.t.cdf(t_score, df = n-1)`
    - greater: $H_a:\mu_1>\mu_2$ `p_value = 1-stats.t.cdf(t_score,df)` or [`stats.t.sf](http://stats.t.sf/)(t_score,df)`
    - two-sided: $H_a:\mu_1\neq \mu_2$ `p_value = 2* stats.t.cdf(-abs(t_score), df = n-1)`

- **Conclusion**
    - Since the p-value `0.006` is smaller than the significance level *α*=0.05, we reject the null hypothesis.
        - **Interpretation**: There is sufficient evidence to support that ..
    - Since the p -value `xxx` is larger than alpha = 0.05 (suggesting the evidence looks normal under the null ,fail to reject the null hypothesis and do not support the alternative
        - **Interpretation**: we don't have sufficient evidence to support Ha. We can't conclude that …

## Others

- Z-test
    - (an approximation of t-test, not required since we can compute the more accurate p-value using t-test)
- F-test (ANOVA):
    - Testing multiple constraints
    - F-score
    - P-value is always right tail

### **Binomial test**

- A Set of 0/1s or knowing # success and # failures (trials)
- Test whether p>,< , != a
- click conversion / product quality / customer interest
- `scipy.stats.binomtest(k, n, p=0.5, alternative=‘two-sided’)` #k:true success num; n: total experiment num; p:expectation prob

```python
#H0: p = 0.2
#Ha: p!= 0.2
## if p= 0.2, X~Binom(n=100, p=0.2)
stats.binom.pmf(29,n=100, p =0.2)

# p -value 
#(extreme cases: cases happen when h0 is false while ha is true)
## if ha is true, p can be large or small, giving large or small x 
#include all x values whose pmf is at most PMF(X=29)

x = np.arange(0, 101)
PMF = stats.binom.pmf(x, n=100, p=0.2)
prob_29 = PMF[29]
p_value = np.sum(PMF[PMF <= prob_29])

# Simple Version
result = binomtest(k=k_success, n=n_trials, p=p_hypothesized, alternative='two-sided')
p_value = result.pvalue
```

- Expectation: able to compute the p-value explicitly (manually) and by calling testing_function

### **Fisher’s exact test (hypergeometric test)**

- 2 sets of samples both containing a mix of 0/1s
- # of successes and # of trials from two groups
- Test whether p1 > p2, p1<p2, or p1!= p2

*Example*

<aside>
📏

 A/B test, a random sample of 60 users is shown Product A, and 20 of them indicate that they like it. Another independent sample of 60 users is shown Product B, and 28 of them report that they like it.

</aside>

```python
import scipy.stats as stats

# 1. param
M = 120  # Total population: 60 + 60
n = 60   # Group A num
N = 48   # Total Successes: 20 + 28
k = 20   # Observed Success in Group A

# support() 
lower, upper = stats.hypergeom.support(M, n, N)
x = np.arange(lower, upper + 1)
PMF = stats.hypergeom.pmf(x, M, n, N)

prob_20 = stats.hypergeom.pmf(20, M, n, N)

# Likelihood Method
p_value = np.sum(PMF[PMF <= prob_20])
p_value
```

```python
#Simple Version
#Contingency Table
#           Like    Dislike
# Product A   20      40    (60-20)
# Product B   28      32    (60-28)

Table = [[20, 40], 
         [28, 32]]

stats.fisher_exact(Table, alternative="two-sided")
```

# 8. Linear Regression

- Run the regression using ols
    
    ```python
    model = ols("y~x1+x2", data).fit()
    model.params ## gives beta hats 
    mdoel.bse ## gives standard error of beta
    #higher , beta_hat less reliable
    ```
    
- Interpret the impact of x on y based on the estimated parameter and the associated p-value
- Be able to perform **prediction** and get the **confidence interval** and **prediction interval, R2**
- $R^2$
    
    ```python
    # ols_model = sm.OLS(Y, X).fit() 
    
    R_squared = ols_model.rsquared
    adj_R_squared = ols_model.rsquared_ad
    ```
    
    ### Prediction $\hat{Y}$
    
    ```python
    new_X_value = 50.0 
    
    X_new = pd.DataFrame({'const': [1], 'advertisement': [new_X_value]})
    
    Y = ols_model.predict(X_new)
    ```
    
- Be able to perform inference (confidence interval, t-test, f-test)
    - **Confidence interval**
        
        $\hat{\beta}_{k} \pm t_{{n-p-1,\alpha/2}}\sqrt{\widehat{\mathrm{Var}(\hat{\beta}_k)}}$
        
        `model.conf_int(alpha=0.05)`
        
        ```python
        ## CI 
        #df = n - p - 1
        df = 34-1-1
        alpha = 0.05
        t_statistics = stats.t.ppf(1-alpha/2,df=df)
        b1_estimated = model.params.iloc[1]
        b1_se = model.bse.iloc[1]
        b1_estimated - t_statistics*b1_se , b1_estimated + t_statistics*b1_se 
        ```
        
        ```python
        ci_table = model.conf_int(alpha=0.05)
        ```
        
    - **test effectivess of coupon**
        - $H_0: \beta_1 = 0$
        - $H_a: \beta_1 \neq 0$
        
        $$
        t = \frac{\hat{\beta_1} - \beta_1}{\sqrt{\text{Var}(\hat{\beta_1})}}
        $$
        
        follows $t_{n-p-1}(0,1)$
        
        `model1.t_test("constraints")`
        
        for example
        
        `model1.t_test("groupon0=0")`
        
    - 
    
    ### overall impact on the revenue
    
    $$
    
    ⁍
    $$
    
    $$
    H_a:  \beta_{\text{groupon}0} + \beta_{\text{groupon}1} + \beta_{\text{groupon}2} + \beta_{\text{groupon}3} \neq 0
    $$
    
    $$
    t = \frac{g(\hat{\beta_0},\hat{\beta_1},\hat{\beta_2}\dots ) - g(\beta_0,\beta_1,\beta_2\dots )}{\sqrt{\text{Var}(g(\hat{\beta_0},\hat{\beta_1},\hat{\beta_2}\dots )}}.
    $$
    
    follows $t_{n-p-1}(0,1)$
    
    $$
    \text{Var}(g(\hat{\beta_0}, \hat{\beta_1}, \dots)) \approx \nabla g^T \Sigma \nabla g 
    $$
    

```python
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats
model2 = smf.ols("revenue ~ groupon0 + groupon1 + groupon2 + groupon3 + timetrend + winter + summer", data=Groupon).fit()

# 2. Estimated parameters, Beta hat
#Intercept, groupon0, groupon1, groupon2, groupon3, timetrend, winter, summer
print(model2.params)

# 3. g_hat
# g_hat = β₀ + β₁ + β₂ + β₃
g_hat = np.sum(model2.params.iloc[1:5])
```

📐  Delta 

```python
#  Σ (Variance-Covariance Matrix)
beta_cov = model2.cov_params() 
print(beta_cov)

# 5. g_dev: Gradient Vector
# g = 1*groupon0 + 1*groupon1 + 1*groupon2 + 1*groupon3 + 0*...
g_dev = np.array([0, 1, 1, 1, 1, 0, 0, 0])

# 6. g_var
#  Var(ĝ) = g_devᵀ * Σ * g_dev
# @  Python/Numpy = Matrix Multiplication
g_var = g_dev @ beta_cov @ (g_dev.T)

# 7. g_se
# Standard Error (SE) = √Var(ĝ)
g_se = g_var**0.5 
```

🧪 Test

```python
alpha = 0.05
# assumption df = N - k - 1 = 34 - 7 - 1 = 26
df_value = 34 - 7 - 1  

g_true = 0 
t_score = (g_hat - g_true) / g_se 
# two-sided P-value
p_value = 2 * stats.t.cdf(-abs(t_score), df=df_value)
```

### $F$ Test / ANOVA

- $H_0:\beta_{\text{groupon0}} = \beta_{\text{groupon1}} = \beta_{\text{groupon2}} = \beta_{\text{groupon3}} = 0$

$F = \frac{(RSS_R - RSS_{UR}) / q}{RSS_{UR} / (N - k - 1)}$