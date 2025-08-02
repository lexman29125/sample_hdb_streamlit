# Exploratory Data Analysis (EDA) Instructions

## 1. Environment Setup

- Ensure Python 3.10+ is installed.
- Recommended: Use a virtual environment (e.g. `conda`).
- Install required libraries:
  ```bash
  pip install numpy pandas matplotlib seaborn
  ```
- For reproducible analysis, use Jupyter Notebook or VSCode Python Interactive.

## 2. Data Loading

- Load data using `pandas.read_csv()` or appropriate function for your format.
- Document data source and loading code for reproducibility.
- Example:
  ```python
  import pandas as pd
  df = pd.read_csv('your_data.csv')
  ```

## 3. Basic Dataset Information

- Display sample rows: `df.head()`, `df.tail()`
- Check shape: `df.shape`
- Data types: `df.info()`
- Missing values: `df.isnull().sum()`
- Duplicated values: `df.duplicated().sum()`
- Basic statistics (numerical): `df.describe()`
- Distribution (categorical): `df[column].value_counts(normalize=True)`

## 4. Data Cleaning & Preparation

- Handle missing values: drop, impute, or flag (`df.dropna()`, `df.fillna()`)
- Remove duplicates: `df.drop_duplicates()`
- Convert data types as needed (e.g., categorical, datetime)
- Feature engineering: create new columns if relevant
- Document all cleaning steps for reproducibility

## 5. Univariate Analysis

### Numerical Variables

- Histogram with KDE: `seaborn.histplot(df[column], kde=True)`
- Summary statistics: `df[column].describe()`, `df[column].mode()`

### Categorical Variables

- Barplot: `seaborn.countplot(x=df[column])`
- Percentage distribution: `df[column].value_counts(normalize=True)`

## 6. Bivariate/Multivariate Analysis

- Scatter plots: `seaborn.pairplot(df[numerical_columns])`
- Correlation matrix: `df.corr()`, `seaborn.heatmap(df.corr(), annot=True)`
- Boxplots: `seaborn.boxplot(x=df[categorical], y=df[numerical])`
- Consider stratifying by relevant categorical variables

## 7. Group Analysis

- Bin numerical variables: `pd.cut()` or `pd.qcut()`
- Grouped barplots/boxplots: `seaborn.boxplot(x=group, y=target)`
- Summary statistics by group: `df.groupby(group).describe()`
- Layer categorical variables where relevant

## 8. Segmentation Analysis

- If clusters are observed, use unsupervised methods (e.g., KMeans, DBSCAN)
- Visualize clusters: scatter plot colored by cluster label
- Summarize segment characteristics: count, percentage, mean/range of variables

## 9. Visualization Best Practices

- Use clear labels, titles, legends
- Choose appropriate color palettes (`seaborn.color_palette()`)
- Save figures for reporting: `plt.savefig('figure.png')`

## 10. Documentation, Reproducibility & Streamlit Output

- Comment code and document each step
- Save cleaned datasets and analysis scripts
- List library versions for reproducibility (`pip freeze > requirements.txt`)
- **Present your final analysis and visualizations using a Streamlit app.**
  - Build an interactive dashboard to showcase key findings, visualizations, and recommendations.
  - Ensure the Streamlit app is user-friendly and well-documented.
  - Example to run: `streamlit run your_app.py`

## 11. Validation Steps

- After each major step, check for:
  - Syntax errors
  - Broken or orphaned columns
  - Consistency of data types
  - Expected output and plots
- Use assertions and visual checks to confirm results

---

## References & Further Reading

- [Real Python: Using pandas and Python to Explore Your Dataset](https://realpython.com/pandas-python-explore-dataset/)
- [Seaborn User Guide](https://seaborn.pydata.org/tutorial.html)
- [DataCamp: Exploratory Data Analysis in Python](https://www.datacamp.com/tutorial/exploratory-data-analysis-python)

---

**Success Criteria:**

- All steps completed and documented
- Clean, well-structured code and outputs
- Visualizations are clear and informative
- Analysis is reproducible and validated
- A working Streamlit app is provided for interactive exploration and reporting of results
