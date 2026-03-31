import streamlit as st


# =========================================================
# 0. PREPROCESSING PHILOSOPHY
# =========================================================
def render_preprocessing_intro():
    st.markdown("## 🧠 Before You Start: Preprocessing Philosophy")
    st.info(
        """
Preprocessing is not just about "cleaning data".

It is about preparing the **geometry of the dataset** so that machine learning models can learn effectively.

Think of your dataset as a **feature space**:
- each row is a point
- each column is a dimension

Every preprocessing step:
- moves points
- reshapes distributions
- changes distances between observations

Your goal is not just to fix errors, but to make patterns clearer and learning more stable.
"""
    )


# =========================================================
# 1. DATA UNDERSTANDING
# =========================================================
def render_data_understanding_theory():
    with st.expander('📖 The "Why" Behind Data Understanding'):
        st.markdown(
            r"""
### 🔍 Diagnosis Before Treatment
Think of this as an **X-ray**. You cannot "fix" data you haven't diagnosed. Understanding the **Statistical DNA** prevents you from applying the wrong transformations.

### 📐 Feature Space Perspective
The "shape" of your data determines how a model "sees" the world:
- **Skewness:** A "stretched" axis that forces models to ignore the long tail.
- **Correlation:** Redundant dimensions that add noise without adding information.
- **Missingness:** A "blind spot" in your feature space that breaks geometric calculations.



### 🎯 The "So What?"
- **Linear/Distance Models:** Sensitive to scale and skewness.
- **Tree-Based Models:** Robust to scale but sensitive to high-cardinality noise.

**Rule of Thumb:** If you don't visualize the distribution, your preprocessing is just guesswork.
"""
        )
        


# =========================================================
# 2. MISSING VALUES
# =========================================================
def render_missing_values_theory():
    with st.expander("📖 Handling Missing Values: Theory & Geometry"):
        st.markdown(
            r"""
### 🌌 The Geometry of "Missingness"
In a $d$-dimensional feature space, an observation is a point $\mathbf{x} = [x_1, x_2, \dots, x_d]$. 
When $x_j$ is missing, the point is no longer a "position"—it becomes a **line** or a **plane** extending infinitely along that axis. Most algorithms (SVM, KNN, Linear Regression) require a single coordinate to calculate distances or gradients.

---

### 🛠 Imputation Strategies

| Method | Mathematical Logic | Best For... |
| :--- | :--- | :--- |
| **Mean** | $x_{j, \text{miss}} = \frac{1}{n} \sum x_j$ | Normal distributions; no outliers. |
| **Median** | $x_{j, \text{miss}} = \text{Sorted}(x_j)[\frac{n}{2}]$ | Skewed data; robust to outliers. |
| **Mode** | $x_{j, \text{miss}} = \text{argmax}(count(x_j))$ | Categorical features (Strings/Labels). |
| **Constant** | $x_{j, \text{miss}} = C$ (e.g., "Unknown") | When "Missing" itself is a signal. |

---

### ⚠️ The "Imputation Trap" (For Students)
Every time you impute, you **reduce variance**. 
- **The Risk:** If you fill 30% of a column with the Mean, your distribution develops a "spike." This can lead the model to over-rely on that average value, potentially causing **overfitting** or biased predictions.
- **The Fix:** Always visualize the distribution (Histogram/KDE) before and after imputation to ensure the "shape" of your data hasn't been distorted.



### 🎯 Selection Heuristic
1. **Is it Categorical?** $\rightarrow$ Use **Mode** or a new category "Missing".
2. **Is it Numerical + Normal?** $\rightarrow$ Use **Mean**.
3. **Is it Numerical + Skewed?** $\rightarrow$ Use **Median**.
4. **Is it >50% Missing?** $\rightarrow$ Consider **dropping** the feature entirely.
"""
        )


# =========================================================
# 2B. CONSISTENCY CHECKS
# =========================================================
def render_consistency_checks_theory():
    with st.expander("📖 Consistency: The 'Language' of Data"):
        st.markdown(
            r"""
### 🧠 Intuition
A model is a mathematical engine. If one row says `10` and another says `"10 USD"`, the engine cannot perform addition, scaling, or comparison. **Consistency is about ensuring every cell in a column speaks the same language.**

### 📐 Feature Space Perspective
Inconsistent formats "break" the axis.
- **The Problem:** A single real-world value (e.g., $100$) encoded in different ways (e.g., `100`, `"100"`, `"$100"`) appears as three distinct, incomparable points.
- **The Result:** The model fails to recognize patterns, treated as noise or "hidden" missing values.

### 🎯 Why It Matters
- **Prevents Crashes:** Most scaling (Min-Max) and encoding functions fail on mixed types.
- **Reliable Patterns:** Ensures the model "sees" the same value the same way every time.
- **Data Integrity:** Prevents "Price" from being ignored because it was accidentally read as "Text."

### ⚠️ Pro-Tip
Always re-run a **Missing Value Check** immediately after casting types. Converting "N/A" strings to floats often creates new `NaN` values.
"""
        )


# =========================================================
# 2C. DUPLICATE REMOVAL
# =========================================================
def render_duplicate_removal_theory():
    with st.expander("📖 Why remove Duplicates?"):
        st.markdown(
            r"""
### 🧠 Intuition

Duplicate rows can cause some observations to be over-represented.

### 📐 Feature Space Perspective

Duplicates place multiple identical points at the same location in feature space, which can bias:
- density
- frequency
- summary statistics

### 🎯 Why it matters

Duplicates may:
- distort distributions
- bias model training
- create misleading patterns

### ⚠️ Trade-offs

Not all duplicates are errors.
For example, repeated transactions or repeated measurements may be meaningful in context.
"""
        )


# =========================================================
# 3. OUTLIERS
# =========================================================
def render_outlier_theory():
    with st.expander("📖 Outlier Detection & Handling (Theory + Math)"):
        st.markdown(
            r"""
### 🧠 Intuition

Outliers are observations that lie unusually far from the typical range of values.

### 📐 Mathematical Definition (IQR rule)

Let:
- $Q_1$ = first quartile
- $Q_3$ = third quartile
- $\mathrm{IQR} = Q_3 - Q_1$

Then a point is often treated as an outlier if:

$$
x < Q_1 - 1.5 \cdot \mathrm{IQR}
\quad \text{or} \quad
x > Q_3 + 1.5 \cdot \mathrm{IQR}
$$

### 📍 Feature Space Impact

Outliers stretch the space and can distort:
- distances
- means
- variances
- optimization behavior

### 🎯 Why it matters

Outliers can strongly affect:
- KNN distance calculations
- standardization
- linear models

### ⚠️ Trade-offs

Removing outliers may improve robustness, but some outliers are genuine and informative.
"""
        )


# =========================================================
# 4. ENCODING
# =========================================================
def render_encoding_theory():
    with st.expander("📖 Encoding: Mapping Labels to Coordinates"):
        st.markdown(
            r"""
### 🧠 Intuition

Computers don't understand "Red," "Green," or "Blue." They understand magnitude and direction. Encoding is the process of translating human labels into a **vector representation**.

### 📐 The Geometry of Choice

#### 1. Label/Ordinal Encoding
Assigns each category a unique integer (e.g., $1, 2, 3$).
- **The Geometry:** It places categories on a single axis at specific intervals.
- **The Risk:** It implies a **natural order**. If you encode "Apple=1, Banana=2, Cherry=3," the model thinks a Cherry is "greater than" an Apple and that the distance between them is exactly 2 units.
- **Best For:** Ordinal data like "Small, Medium, Large."



#### 2. One-Hot Encoding
Creates a new "binary" column for each category.
- **The Geometry:** It moves each category into its own **independent dimension**. Every category becomes a unit vector:
  - Red: $(1, 0, 0)$
  - Green: $(0, 1, 0)$
  - Blue: $(0, 0, 1)$
- **The Benefit:** All categories are equidistant ($d = \sqrt{2}$) from each other. No category is "greater" than another.
- **Best For:** Nominal data with no inherent order (e.g., Countries, Colors).



### 🎯 The "Curse of Dimensionality"
Students should be wary: If a feature has 100 unique categories, One-Hot Encoding adds 100 new dimensions. This can make the feature space "sparse," making it harder for models to find patterns—a phenomenon known as the **Curse of Dimensionality**.

### 📌 Summary Table

| Technique | Logic | Mathematical Implication |
| :--- | :--- | :--- |
| **Ordinal** | $x \in \{1, 2, 3, \dots\}$ | Imposes a linear rank/order. |
| **One-Hot** | $x \in \{0, 1\}^k$ | Ensures equidistance in $k$-dimensions. |
| **Target** | $x = E[y|category]$ | Compresses category into its relationship with the target. |
"""
        )


# =========================================================
# 5. FEATURE SCALING / NORMALIZATION
# =========================================================
def render_scaling_theory():
    with st.expander("📖 Feature Scaling & Normalization (Theory + Math)"):
        st.markdown(
            r"""
### 🧠 Intuition

Different features may exist on very different scales.

For example:
- income may range from 1,000 to 10,000
- GPA may range from 0 to 5

Without scaling, larger-scale features dominate.

A useful analogy:
**scaling is like converting different currencies into the same unit so fair comparison becomes possible.**

### 📐 Min-Max Normalization

$$
\tilde{x} = \frac{x - \min(x)}{\max(x) - \min(x)}
$$

This maps values into the range $[0, 1]$.

### 📍 Feature Space Impact

Normalization rescales axes so that values occupy a common bounded range.

This changes the geometry of the data by compressing or stretching dimensions.

### 🎯 Why it matters

This is especially important for:
- KNN
- SVM
- neural networks
- gradient-based methods

If skipped, large-scale features may dominate model behavior.

### ⚠️ Trade-offs

Min-max normalization is sensitive to outliers. Extreme values can compress the majority of the data into a narrow range.
"""
        )


# =========================================================
# 6. STANDARDIZATION / VARIANCE SCALING
# =========================================================
def render_standardization_theory():
    with st.expander("📖 Variance Scaling / Standardization (Z-Score)"):
        st.markdown(
            r"""
### 🧠 Intuition

Standardization makes features comparable by centering them and adjusting their spread.

A helpful analogy:
**it is like converting exam results from different classes into standard scores, so performance can be compared fairly.**

### 📐 Mathematical Definition

$$
z = \frac{x - \mu}{\sigma}
$$

Where:
- $\mu$ is the feature mean
- $\sigma$ is the feature standard deviation

### 📍 Feature Space Impact

Standardization:
- centers the feature around 0
- makes the variance approximately 1

In feature space, this rebalances axes so that one dimension does not dominate another simply because of scale.

### 🎯 Why it matters

This is essential for undergraduates to understand because many common models are scale-sensitive:

- **KNN** uses distances directly
- **SVM** constructs margins based on geometry
- **gradient-based optimization** can behave poorly if features are badly scaled

If skipped, the model may treat large-scale features as artificially more important.

### ⚠️ Trade-offs

Standardization does not remove skewness. If data is highly skewed, a power transform may still be needed.
"""
        )


# =========================================================
# 7. POWER TRANSFORM
# =========================================================
def render_power_transform_theory():
    with st.expander("📖 Power Transformation (Yeo-Johnson)"):
        st.markdown(
            r"""
### 🧠 Intuition

Some numeric features are heavily skewed. Power transforms reshape them into a more symmetric form.

### 📐 Yeo-Johnson Transformation

Yeo-Johnson is a flexible family of transformations that can handle zero and negative values.

In simplified form:

$$
x^{(\lambda)} =
\begin{cases}
\frac{(x+1)^\lambda - 1}{\lambda}, & x \ge 0, \lambda \ne 0 \\
\log(x+1), & x \ge 0, \lambda = 0
\end{cases}
$$

with a corresponding definition for negative values.

### 📍 Feature Space Impact

Power transforms:
- compress extreme values
- reduce skew
- reshape the cloud of points into a more balanced form

### 🎯 Why it matters

Skewed features can negatively affect:
- linear models
- distance-based methods
- standardization quality

### ⚠️ Trade-offs

Power transforms improve geometry for learning, but reduce the direct interpretability of the original scale.
"""
        )


# =========================================================
# 8. FINAL REVIEW
# =========================================================
def render_review_theory():
    with st.expander("📖 Why review the Dataset after Processing?"):
        st.markdown(
            r"""
### 🧠 Intuition

Every preprocessing step changes the data. Review ensures the changes were useful rather than harmful.

### 📍 What to check

- Have distributions become more reasonable?
- Did correlations change substantially?
- Was too much data removed?

### 🎯 Why it matters

Preprocessing should improve the learnability of the feature space, not destroy signal.

### 📌 Key takeaway

Always validate preprocessing visually and statistically.
"""
        )


# =========================================================
# 9. EXPORT
# =========================================================
def render_export_theory():
    with st.expander("📖 Why export the Processed Dataset?"):
        st.markdown(
            r"""
### 🧠 Intuition

The processed dataset is the output of your preprocessing workflow.

### 🎯 Why it matters

You may now use it for:
- model training
- downstream feature analysis
- reproducible experiments

### 📌 Best practice

Keep the raw dataset unchanged and export processed versions separately so your workflow remains reproducible.
"""
        )