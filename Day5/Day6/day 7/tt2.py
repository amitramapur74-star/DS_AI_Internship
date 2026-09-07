import pandas as pd

# Create student performance dataset
data = {
    "Name": [
        "Asha", "Rahul", "Sneha", "Arjun", "Priya",
        "Kiran", "Neha", "Vijay", "Anu", "Ravi",
        "Asha", "Rahul", "Meena", "Karthik", "Divya"
    ],
    "Maths": [
        85, 72, None, 90, 65,
        78, 88, None, 70, 95,
        85, 72, 82, None, 91
    ],
    "Science": [
        80, None, 75, 92, 60,
        85, 90, 68, None, 88,
        80, None, 79, 84, 93
    ],
    "English": [
        78, 75, 80, None, 70,
        82, 89, 65, 72, 90,
        78, 75, 81, 86, None
    ]
}

df = pd.DataFrame(data)

# Display original dataset
print("Original Dataset:")
print(df)

# Find shape
print("\nOriginal Shape:")
print(df.shape)

# Identify missing values
print("\nMissing Values:")
print(df.isnull())

# Count missing values
print("\nCount of Missing Values:")
print(df.isnull().sum())

# Count duplicate rows
print("\nNumber of Duplicate Rows:")
print(df.duplicated().sum())

# Remove duplicate rows
df = df.drop_duplicates()

print("\nDataset After Removing Duplicates:")
print(df)

# Handle missing values using column mean
df["Maths"] = df["Maths"].fillna(df["Maths"].mean())
df["Science"] = df["Science"].fillna(df["Science"].mean())
df["English"] = df["English"].fillna(df["English"].mean())

# Check missing values after handling
print("\nMissing Values After Handling:")
print(df.isnull().sum())

# Final cleaned dataset
print("\nCleaned Dataset:")
print(df)

# Find final shape
print("\nCleaned Dataset Shape:")
print(df.shape)
print(df.describe())