import pandas as pd


data = pd.read_csv("student.csv")

print("Original Dataset:")
print(data)

print("\nOriginal Shape:")
print(data.shape)

print("\nMissing Values:")
print(data.isnull())

print("\nCount of Missing Values:")
print(data.isnull().sum())

print("\nNumber of Duplicate Rows:")
print(data.duplicated().sum())

data = data.drop_duplicates()

print("\nDataset After Removing Duplicates:")
print(data)

data["Maths"] = data["Maths"].fillna(data["Maths"].mean()).astype(int)
data["Science"] = data["Science"].fillna(data["Science"].mean()).astype(int)
data["English"] = data["English"].fillna(data["English"].mean()).astype(int)

print("\nMissing Values After Handling:")
print(data.isnull().sum())

print("\nCleaned Dataset:")
print(data)

print("\nCleaned Dataset Shape:")
print(data.shape)