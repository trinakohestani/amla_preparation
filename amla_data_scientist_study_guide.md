# AMLA Data Scientist Test: Detailed Review (Statistics + Python)

Test format: 10 MCQs on statistics, 20 MCQs on Python (or R).
Tip: the AML context favors topics like class imbalance, base rates, false positives/negatives, anomaly detection and model explainability.

---

# PART 1: STATISTICS

## 1. Descriptive Statistics

**Measures of location**
- **Mean:** x̄ = Σxᵢ / n. Sensitive to outliers.
- **Median:** middle value of the sorted data. Robust.
- **Mode:** most frequent value.
- **Skewness:** right-skewed (positive) means mode < median < mean. Left-skewed is the reverse. Transaction amounts are typically right-skewed.

**Measures of spread**
- **Variance:** s² = Σ(xᵢ − x̄)² / (n−1) for a sample, σ² = Σ(xᵢ − μ)² / N for a population (n−1 is Bessel's correction).
- **Standard deviation:** √variance, in the units of the data.
- **IQR** = Q3 − Q1. **Boxplot outlier rule:** values outside [Q1 − 1.5·IQR, Q3 + 1.5·IQR].
- **Coefficient of variation:** σ/μ, useful for comparing spread across different scales.

**Measures of association**
- **Covariance:** direction of the relationship, depends on scale.
- **Pearson correlation r:** measures *linear* association, −1 ≤ r ≤ 1. Sensitive to outliers.
- **Spearman:** rank-based, captures monotonic relationships, more robust.
- **Correlation ≠ causation.** r = 0 only means "no linear relationship", not "independent".

---

## 2. Probability

**Basic rules**
- P(A ∪ B) = P(A) + P(B) − P(A ∩ B)
- Conditional probability: P(A|B) = P(A ∩ B) / P(B)
- Independence: P(A ∩ B) = P(A)·P(B)
- Law of total probability: P(B) = Σ P(B|Aᵢ)·P(Aᵢ)

**Bayes' theorem**

P(A|B) = P(B|A)·P(A) / P(B)

*AML example:* 0.5% of all transactions are suspicious. The system flags 90% of the suspicious ones (sensitivity) but also 5% of the legitimate ones (false positive rate).

- P(alert) = 0.9·0.005 + 0.05·0.995 = 0.0045 + 0.04975 = 0.05425
- P(suspicious | alert) = 0.0045 / 0.05425 ≈ **8.3%**

Despite good sensitivity, over 90% of alerts are false alarms. This is the **base rate fallacy**.


Core Performance Metrics

**• Sensitivity (True Positive Rate / Recall):**
	• Definition: The proportion of actual positives that are correctly identified.
	• Formula: \(\text{Sensitivity} = \frac{\text{TP}}{\text{TP} + \text{FN}}\)
	• Use case: High sensitivity means very few false negatives; ideal for screening tests (SnN-OUT: a sensitive test with a negative result rules out the condition).
 
**• Specificity (True Negative Rate):**
	• Definition: The proportion of actual negatives that are correctly identified.
	• Formula: \(\text{Specificity} = \text{TN} / (\text{TN} + \text{FP}) = \frac{\text{TN}}{\text{FP} + \text{TN}}\)
	• Use case: High specificity means very few false positives; ideal for confirmatory tests (SpP-IN: a specific test with a positive result rules in the condition).
 
**• False Positive Rate (FPR):**
	• Definition: The proportion of actual negatives that are wrongly called positive (1 - Specificity).
	• Formula: \(\text{FPR} = \frac{\text{FP}}{\text{FP} + \text{TN}}\)
 
**• False Negative Rate (FNR):**
	• Definition: The proportion of actual positives that are wrongly called negative (1 - Sensitivity).
	• Formula: \(\text{FNR} = \frac{\text{FN}}{\text{TP} + \text{FN}}\)

**Key distributions**

| Distribution | Use case | Parameters | E[X] / Var |
|---|---|---|---|
| Bernoulli | one trial, success/failure | p | p / p(1−p) |
| Binomial | number of successes in n trials | n, p | np / np(1−p) |
| Poisson | rare events per interval | λ | λ / λ |
| Normal | measurement errors, sums of many effects | μ, σ | μ / σ² |
| Exponential | waiting time between events | λ | 1/λ / 1/λ² |
| Uniform | all values equally likely | a, b | (a+b)/2 / (b−a)²/12 |
| Log-normal | amounts, incomes (right-skewed) | μ, σ | - |

- **z-transformation:** z = (x − μ)/σ. Afterwards N(0,1) applies.
- Normal distribution: ~68% of values within ±1σ, ~95% within ±2σ, ~99.7% within ±3σ.
- **Central Limit Theorem (CLT):** the mean of many independent observations is approximately normally distributed, regardless of the original distribution (for large enough n, rule of thumb n ≥ 30).
- **Law of large numbers:** the sample mean converges to the expected value.
- **Standard error of the mean:** SE = σ/√n. It shrinks as n grows, but the standard deviation of the data does not.

---

## 3. Estimation and Confidence Intervals

- **Point estimator:** e.g. x̄ for μ. A good estimator is **unbiased**, **consistent** and **efficient**.
- **95% confidence interval** for the mean: x̄ ± t · s/√n (or ± 1.96 · σ/√n if σ is known).
- **Correct interpretation:** if the procedure is repeated many times, 95% of the computed intervals contain the true value.
  **Wrong:** "There is a 95% probability that the true value lies in this interval."
- Larger n or lower confidence level gives a narrower interval.
- **Maximum Likelihood (MLE):** choose the parameters that maximize the probability of the observed data.

---

## 4. Hypothesis Testing

**Procedure:** state H₀ and H₁, choose significance level α (usually 0.05), compute the test statistic, determine the p-value, decide (p < α → reject H₀).

**Error types**

| | H₀ true | H₀ false |
|---|---|---|
| H₀ rejected | **Type I error (α)**, false positive | correct (power = 1−β) |
| H₀ retained | correct | **Type II error (β)**, false negative |

- **p-value:** P(data at least this extreme | H₀ true). It is **not** P(H₀ is true) and **not** the effect size.
- Statistical significance ≠ practical relevance. With huge n, even tiny effects become significant. Report effect sizes (e.g. Cohen's d).
- **Power** increases with larger n, larger effect size and higher α.
- **One-sided vs. two-sided:** one-sided only if the direction is justified in advance.

**Which test when?**

| Situation | Test |
|---|---|
| Mean of one group vs. fixed value | One-sample t-test |
| 2 independent groups, metric, roughly normal | Two-sample t-test (Welch if variances differ) |
| Before/after on the same units | Paired t-test |
| 2 groups, not normal | Mann-Whitney U |
| Paired, not normal | Wilcoxon signed-rank |
| > 2 groups, metric | ANOVA (non-parametric: Kruskal-Wallis) |
| 2 categorical variables, independence | Chi-square test |
| Check normality | Shapiro-Wilk, QQ plot |
| Compare distributions | Kolmogorov-Smirnov |

**Multiple testing:** with m tests, the probability of at least one false alarm rises to 1 − (1−α)^m. Remedies: **Bonferroni** (α/m, conservative) or **Benjamini-Hochberg** (controls the false discovery rate).

---

## 5. Linear Regression

y = β₀ + β₁x₁ + … + βₖxₖ + ε, estimated with **OLS** (minimizes the sum of squared residuals).

**Assumptions (LINE)**
1. **L**inearity of the relationship
2. **I**ndependence of errors (no autocorrelation, Durbin-Watson)
3. **N**ormally distributed residuals (matters for tests and intervals)
4. **E**qual variance, homoscedasticity (otherwise heteroscedasticity, Breusch-Pagan)

Additionally: **no perfect multicollinearity**.

**Interpretation:** βⱼ is the expected change in y for +1 in xⱼ, *holding all other variables constant* (ceteris paribus).

**Goodness of fit**
- **R²** = share of variance explained. It increases with every added variable.
- **Adjusted R²** penalizes unnecessary variables.
- **RMSE / MAE:** error in units of y. RMSE penalizes large errors more.
- **AIC / BIC:** model comparison (lower is better, BIC penalizes complexity more).

**Problems**
- **Multicollinearity:** strongly correlated predictors lead to unstable coefficients and large standard errors. Detect with **VIF** (> 5-10 is critical).
- **Outliers and leverage points:** Cook's distance.
- **Overfitting:** too many variables, poor generalization.

**Regularization**
- **Ridge (L2):** shrinks coefficients but does not set them to 0. Good for multicollinearity.
- **Lasso (L1):** sets some coefficients to exactly 0 (feature selection).
- **Elastic Net:** a mix of both. **Standardize** features first!

---

## 6. Logistic Regression and Classification

- Models log(p/(1−p)) = β₀ + β₁x₁ + … (log-odds). The output is a probability (sigmoid).
- **exp(β)** = odds ratio: factor by which the odds change per +1 in x.
- Estimated via maximum likelihood (not OLS).

**Confusion matrix and metrics**

| | predicted positive | predicted negative |
|---|---|---|
| actually positive | TP | FN |
| actually negative | FP | TN |

- **Accuracy** = (TP+TN)/all. Misleading under imbalance (99% accuracy by always predicting "negative").
- **Precision** = TP/(TP+FP): how many alerts are real?
- **Recall / sensitivity** = TP/(TP+FN): how many real cases are found?
- **Specificity** = TN/(TN+FP)
- **F1** = 2·P·R/(P+R)
- **ROC-AUC:** probability that a random positive case is scored higher than a random negative case. 0.5 = chance.
- **PR-AUC:** more informative than ROC-AUC for strongly imbalanced data.
- The **threshold** is a tuning knob: lowering it gives more recall, less precision.

---

## 7. Model Validation and Machine Learning

- **Bias-variance tradeoff:** simple model = high bias (underfitting), complex model = high variance (overfitting).
- **Train/validation/test:** the test set is used only once at the end.
- **k-fold cross-validation:** more robust estimate. Use **stratified** folds under imbalance and **time-ordered** splits for time series (no random split!).
- **Data leakage:** information from the test set or the future flows into training (e.g. scaling on all data before the split, features that contain the target).
- **Imbalance strategies:** class weights, over-/undersampling, SMOTE (training data only), threshold tuning, appropriate metrics.
- **Random Forest:** many trees on bootstrap samples (bagging), reduces variance. **Gradient Boosting:** trees built sequentially, each corrects the errors of the previous one, reduces bias.
- **Feature importance / SHAP:** explainability, important in the regulated AML environment.

**Unsupervised learning**
- **K-Means:** K must be specified, spherical clusters, scale-sensitive. Choose K via elbow or silhouette.
- **DBSCAN:** density-based, arbitrary shapes, labels noise as −1, no K needed (parameters eps, min_samples).
- **PCA:** orthogonal directions of maximum variance, standardize first.
- **Isolation Forest:** anomalies can be isolated with few random splits, so short paths = suspicious.

---

# PART 2: PYTHON

## 1. Basics and Pitfalls

```python
# Data types
type(3), type(3.0), type("a"), type([1]), type((1,)), type({1}), type({"a": 1})

# Mutable vs. immutable
a = [1, 2, 3]
b = a            # same reference!
b.append(4)      # a is now [1, 2, 3, 4]
c = a.copy()     # real (shallow) copy

# Slicing: [start:stop:step], stop is exclusive
x = [0, 1, 2, 3, 4, 5]
x[1:4]    # [1, 2, 3]
x[-2:]    # [4, 5]
x[::2]    # [0, 2, 4]
x[::-1]   # reversed

# Comparison
[1, 2] == [1, 2]   # True (value)
[1, 2] is [1, 2]   # False (identity)
```

- **Mutable:** list, dict, set. **Immutable:** int, float, str, tuple, frozenset.
- **Mutable default argument:**
  ```python
  def f(x, lst=[]):      # trap: lst is created only once
      lst.append(x); return lst
  f(1); f(2)             # [1], then [1, 2]
  ```
- `//` is floor division, `%` modulo, `**` power. `-7 // 2` gives `-4` (rounds down!).
- Truthiness: `0`, `""`, `[]`, `{}`, `None` are `False`.
- Float imprecision: `0.1 + 0.2 == 0.3` is `False`. Use `math.isclose`.

**Comprehensions and generators**
```python
[x**2 for x in range(5) if x % 2 == 0]        # [0, 4, 16]
{k: v for k, v in zip("abc", [1, 2, 3])}      # {'a': 1, 'b': 2, 'c': 3}
{x % 3 for x in range(10)}                    # {0, 1, 2}
sum(x for x in range(5))                      # generator, memory-efficient
```

**Dictionaries and sets**
```python
d = {"a": 1}
d.get("b", 0)             # 0 (no KeyError)
d["b"]                    # KeyError!
d.items(); d.keys(); d.values()
{1, 2} | {2, 3}   # {1, 2, 3}  (union)
{1, 2} & {2, 3}   # {2}        (intersection)
```

**Functions, lambda, error handling**
```python
sorted(data, key=lambda t: t[1], reverse=True)
list(map(lambda x: x * 2, [1, 2, 3]))      # [2, 4, 6]
list(filter(lambda x: x > 1, [1, 2, 3]))   # [2, 3]

def f(*args, **kwargs): ...                 # arbitrary arguments

try:
    risky()
except ValueError as e:
    ...
finally:
    ...
```

---

## 2. NumPy

```python
import numpy as np
a = np.array([[1, 2, 3], [4, 5, 6]])
a.shape                 # (2, 3)
a.mean(axis=0)          # [2.5, 3.5, 4.5]  -> axis=0: across rows, per column
a.mean(axis=1)          # [2., 5.]         -> axis=1: across columns, per row
a[a > 2]                # boolean indexing -> [3, 4, 5, 6]
a.reshape(3, 2); a.T
np.where(a > 2, 1, 0)
```
- **Broadcasting:** arrays of different shapes are automatically aligned (e.g. matrix + row vector).
- Vectorized operations are much faster than Python loops.
- `np.std` uses **ddof=0** (population) by default, whereas `pandas.Series.std` uses **ddof=1** (sample). A popular trap!
- NumPy slices are **views**, not copies (use `.copy()`).

---

## 3. pandas

**Selection**
```python
df["col"]                          # Series
df[["a", "b"]]                     # DataFrame
df.loc[df["amount"] > 1000, ["id", "amount"]]   # label/boolean, end INCLUSIVE
df.iloc[0:3, 1:3]                  # position, end EXCLUSIVE
df.query("amount > 1000 and country == 'DE'")
df[(df.a > 1) & (df.b < 5)]        # & and | with parentheses, not "and"/"or"!
df["x"].isin(["DE", "FR"])
```

**Cleaning and transformation**
```python
df.info(); df.describe(); df.dtypes; df.shape
df.isna().sum()                          # missing values per column
df.dropna(subset=["amount"])
df["amount"].fillna(df["amount"].median())
df.drop_duplicates(subset=["id"], keep="first")
df["date"] = pd.to_datetime(df["date"])
df["month"] = df["date"].dt.month
df["amount"].astype(float)
df.rename(columns={"old": "new"})
df["cat"] = df["x"].map({"a": 1, "b": 2})
df["flag"] = np.where(df["amount"] > 10000, 1, 0)
df["bin"] = pd.cut(df["amount"], bins=[0, 100, 1000, np.inf])    # fixed edges
df["q"] = pd.qcut(df["amount"], q=4)                            # quantiles
```

**Aggregation**
```python
df.groupby("country")["amount"].mean()
df.groupby("country").agg(total=("amount", "sum"),
                          n=("id", "count"),
                          avg=("amount", "mean"))
df.groupby(["country", "type"]).size().reset_index(name="n")
df["rank"] = df.groupby("country")["amount"].rank(ascending=False)
df["cum"] = df.groupby("id")["amount"].cumsum()
df["amount"].rolling(7).mean()             # moving average
df["prev"] = df["amount"].shift(1)         # value of the previous row
df["amount"].value_counts(normalize=True)
df.pivot_table(index="country", columns="type", values="amount", aggfunc="sum")
```
- `count()` counts only **non-NaN** values, `size()` counts all rows.
- `sum()` ignores NaN by default (`skipna=True`).

**Joins and concatenation**
```python
df.merge(other, on="id", how="inner")   # inner, left, right, outer
pd.concat([df1, df2], axis=0)           # stacked vertically
pd.concat([df1, df2], axis=1)           # side by side
```
- **inner:** only matches in both. **left:** all from the left. **outer:** all from both.
- With duplicate keys in `merge`, rows multiply (Cartesian product per key).

**Typical traps**
```python
# Chained assignment: may have no effect
df[df.a > 1]["b"] = 5            # bad
df.loc[df.a > 1, "b"] = 5        # correct

# NaN comparison
np.nan == np.nan                 # False -> use isna()

# inplace: methods usually return a new copy
df.sort_values("a")              # df is unchanged, assign the result!
```
- After `groupby`, the grouping column becomes the index (`reset_index()` or `as_index=False`).
- `apply` is slow, vectorize where possible.
- Use `df.copy()` to create true copies.

---

## 4. scikit-learn

```python
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, precision_recall_curve)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(class_weight="balanced", max_iter=1000))
])
pipe.fit(X_train, y_train)
proba = pipe.predict_proba(X_test)[:, 1]     # probability for class 1
roc_auc_score(y_test, proba)
```

**Key points**
- Consistent API: `fit()` learns, `transform()` applies, `predict()` predicts.
- **Fit the scaler on train only**, use only `transform` on test (otherwise leakage). A `Pipeline` does this correctly automatically, including inside cross-validation.
- `confusion_matrix` has the shape `[[TN, FP], [FN, TP]]`.
- `predict` uses a 0.5 threshold; use `predict_proba` to set custom thresholds.
- Scaling matters for: k-NN, SVM, K-Means, PCA, regularization. Not needed for tree models.
- Categorical variables: `OneHotEncoder` (nominal), `OrdinalEncoder` (ordinal).
- `GridSearchCV` / `RandomizedSearchCV` for hyperparameter tuning with cross-validation.
- Set `random_state` for reproducibility.
- `stratify=y` preserves the class ratio, important under imbalance.

```python
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

iso = IsolationForest(contamination=0.01, random_state=42).fit(X)
iso.predict(X)          # -1 = anomaly, 1 = normal
```

---

## 5. Visualization (Basics)

```python
import matplotlib.pyplot as plt
import seaborn as sns
plt.hist(df["amount"], bins=30)       # distribution
plt.scatter(x, y)                     # relationship
sns.boxplot(x="type", y="amount", data=df)   # spread, outliers
sns.heatmap(df.corr(), annot=True)    # correlation matrix
```
- Histogram for distributions, boxplot for outliers and group comparisons, scatter for relationships, heatmap for correlations.
- For strongly right-skewed amounts, a **log scale** or `np.log1p` helps.

---

# Common Exam Traps (Quick List)

1. The p-value is not the probability that H₀ is true.
2. Confidence interval interpretation (see above).
3. Accuracy on imbalanced data is worthless.
4. Fitting the scaler on test data is data leakage.
5. `loc` is inclusive, `iloc` is exclusive.
6. `np.std` (ddof=0) vs. `pd.std` (ddof=1).
7. `df.sort_values()` without assignment changes nothing.
8. Mutable default arguments, and reference instead of copy with lists.
9. Correlation ≠ causation, r = 0 ≠ independence.
10. R² always rises with more variables, so use adjusted R² or cross-validation.
