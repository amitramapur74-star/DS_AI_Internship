import pandas as pd

# Create a Pandas Series with different letter cases and missing values
names = pd.Series([
    "AMIT",
    "Rahul",
    None,
    "Sam",
    "KIRAN",
    None,
    "Ananya"
])

print("Original Series:")
print(names)

# Detect missing values
print("\nMissing values:")
print(names.isnull())

# Fill missing values
names = names.fillna("Unknown")

print("\nSeries after filling missing values:")
print(names)

# Convert all names to lowercase
names_lower = names.str.lower()

print("\nNames in lowercase:")
print(names_lower)

# Filter names containing the letter 'a'
filtered_names = names_lower[names_lower.str.contains("a")]

print("\nNames containing the letter 'a':")
print(filtered_names)