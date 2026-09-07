import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Improve plot appearance
sns.set(style="whitegrid")

# ----------------------------------------------------------
# STEP 2 — Load / Create Dataset
# (In real projects: df = pd.read_csv("dataset.csv"))
# ----------------------------------------------------------
data = {
    "Age": [25,30,35,40,28,32,45,50,23,36,29,41],
    "Salary": [30000,40000,50000,65000,42000,48000,80000,90000,28000,52000,46000,70000],
    "Experience": [1,3,7,10,2,5,15,20,1,8,4,12],
    "Department": ["IT","HR","IT","Finance","HR","IT","Finance","Finance","HR","IT","HR","Finance"],
    "Gender": ["M","F","M","M","F","F","M","M","F","F","M","F"]
}

df = pd.DataFrame(data)

# ----------------------------------------------------------
# TOPIC 1 — DATASET INSPECTION
# ----------------------------------------------------------

# View first and last rows
print("\nFirst 5 rows:")
print(df.head())

print("\nLast 5 rows:")
print(df.tail())

# Dataset shape
print("\nDataset shape (rows, columns):", df.shape)

# Data types and missing values
print("\nDataset info:")
print(df.info())

# Summary statistics for numerical columns
print("\nSummary statistics:")
print(df.describe())

# ----------------------------------------------------------
# TOPIC 2 — UNIVARIATE ANALYSIS
# Analyze ONE variable at a time
# ----------------------------------------------------------

# HISTOGRAM — Distribution of Age
plt.figure()
sns.histplot(df["Age"], kde=True)
plt.title("Age Distribution")
plt.show()

# HISTOGRAM — Distribution of Salary
plt.figure()
sns.histplot(df["Salary"], kde=True)
plt.title("Salary Distribution")
plt.show()

# BOXPLOT — Detect spread & outliers in Salary
plt.figure()
sns.boxplot(x=df["Salary"])
plt.title("Salary Boxplot")
plt.show()

# CATEGORICAL ANALYSIS — Frequency counts
print("\nDepartment counts:")
print(df["Department"].value_counts())

print("\nGender counts:")
print(df["Gender"].value_counts())

# Bar plot for categorical variable
plt.figure()
sns.countplot(x="Department", data=df)
plt.title("Department Distribution")
plt.show()

# ----------------------------------------------------------
# TOPIC 3 — BIVARIATE ANALYSIS
# Study relationship between TWO variables
# ----------------------------------------------------------

# SCATTER PLOT — Age vs Salary
plt.figure()
sns.scatterplot(x="Age", y="Salary", data=df)
plt.title("Age vs Salary")
plt.show()

# SCATTER PLOT — Experience vs Salary
plt.figure()
sns.scatterplot(x="Experience", y="Salary", data=df)
plt.title("Experience vs Salary")
plt.show()

# BOXPLOT — Salary by Gender
plt.figure()
sns.boxplot(x="Gender", y="Salary", data=df)
plt.title("Salary by Gender")
plt.show()

# BOXPLOT — Salary by Department
plt.figure()
sns.boxplot(x="Department", y="Salary", data=df)
plt.title("Salary by Department")
plt.show()

# ----------------------------------------------------------
# TOPIC 4 — CORRELATION ANALYSIS
# ----------------------------------------------------------

# Correlation matrix (numerical columns only)
corr_matrix = df.corr(numeric_only=True)
print("\nCorrelation Matrix:")
print(corr_matrix)

# HEATMAP — visualize correlations
plt.figure()
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# ----------------------------------------------------------
# TOPIC 5 — OUTLIER DETECTION
# ----------------------------------------------------------

# Boxplot for Age
plt.figure()
sns.boxplot(x=df["Age"])
plt.title("Age Outliers")
plt.show()

# Boxplot for Experience
plt.figure()
sns.boxplot(x=df["Experience"])
plt.title("Experience Outliers")
plt.show()

# ----------------------------------------------------------
# FINAL STEP — DOCUMENT INSIGHTS (PRINT SAMPLE INSIGHTS)
# Students should write their own observations here.
# ----------------------------------------------------------

print("\n===== SAMPLE INSIGHTS =====")
print("1. Salary increases with Experience and Age (positive correlation).")
print("2. Finance department shows higher salary range.")
print("3. No extreme outliers detected in Age or Experience.")
print("4. Gender distribution appears balanced.")
print("5. Experience strongly influences Salary.")