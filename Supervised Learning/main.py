import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
df = pd.read_csv("data.csv")

print("Student Dataset:")
print(df)

# ---------------- SUPERVISED LEARNING ----------------

# Input features
X = df[["Attendance", "Study_Hours", "Assignment_Marks", "Previous_Marks"]]

# Target
y = df["Final_Marks"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

print("\n--- Supervised Learning ---")
print("Actual Marks:", y_test.values)
print("Predicted Marks:", y_pred)

# Model evaluation
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("R2 Score:", r2)


# ---------------- UNSUPERVISED LEARNING ----------------

# Select data for clustering
cluster_data = df[
    ["Attendance", "Study_Hours", "Assignment_Marks", "Previous_Marks"]
]

# Create K-Means model
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)

# Create groups
df["Student_Group"] = kmeans.fit_predict(cluster_data)

print("\n--- Unsupervised Learning ---")
print(df[["Attendance", "Study_Hours",
          "Assignment_Marks", "Previous_Marks",
          "Student_Group"]])
