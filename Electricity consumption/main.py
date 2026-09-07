import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
df = pd.read_csv("electricity_consumption.csv")

print("Dataset:")
print(df)

# Convert Time into numbers
encoder = LabelEncoder()
df["Time"] = encoder.fit_transform(df["Time"])

# Select input features and target
X = df[["Temperature", "Appliances", "Time", "Previous_Usage"]]
y = df["Consumption"]

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Predict electricity consumption
y_pred = model.predict(X_test)

# Display predictions
print("\nActual Consumption:")
print(y_test.values)

print("\nPredicted Consumption:")
print(y_pred)

# Calculate performance
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nMean Absolute Error:", mae)
print("R2 Score:", r2)

# Example prediction
new_data = [[30, 8, encoder.transform(["Evening"])[0], 12]]

prediction = model.predict(new_data)

print("\nPredicted electricity consumption:", prediction[0], "kWh")