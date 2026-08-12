import pandas as pd

data = {
    "CustomerID": [101, 102, 103, 104, 105, 106, 107, 107, 108, 109],
    "Name": ["Amit", "Saran", "John", "None", "Priya", "David", "Meena", "Meena", "Ali", "Riya"],
    "Age": [25, None, 30, 22, None, 28, 35, 35, None, 26],
    "City": [" Bangalore", "Mumbai ", "Delhi", None, "Bangalore", "Chennai", "Mumbai", "Mumbai", "Delhi", " Bangalore "],
    "OrderAmount": [2500, 1800, None, 2200, 3000, None, 1500, 1500, 2700, None],
    "PaymentMethod": ["UPI", "Card", "Cash", "Card", None, "UPI", "Cash", "Cash", "Card", "UPI"],
    "Date": [
        "2024-01-05",
        "2024-01-10",
        "2024-02-01",
        "2024-02-05",
        "2024-03-01",
        "2024-03-05",
        "2024-03-10",
        "2024-03-10",
        "2024-04-01",
        "2024-04-05"
    ]
}

df = pd.DataFrame(data)

print("First rows:\n", df.head())

print("\nDataset info:")
print(df.info())

print("\nMissing values per column:")
print(df.isna().sum())

df["Age"] = df["Age"].fillna(df["Age"].mean())
df["OrderAmount"] = df["OrderAmount"].fillna(df["OrderAmount"].mean())
df["City"] = df["City"].fillna(df["City"].mode()[0])
df["PaymentMethod"] = df["PaymentMethod"].fillna(df["PaymentMethod"].mode()[0])
df["Name"] = df["Name"].fillna("Unknown")

print("\nData types BEFORE conversion:")
print(df.dtypes)

df["Age"] = df["Age"].astype(int)
df["Date"] = pd.to_datetime(df["Date"])

print("\nData types AFTER conversion:")
print(df.dtypes)

df["City"] = df["City"].str.strip()

df["City"] = df["City"].str.lower()

print("\nCity column after cleaning:")
print(df["City"])

print("\nNumber of duplicate rows:")
print(df.duplicated().sum())

df = df.drop_duplicates()

print("\nShape after removing duplicates:", df.shape)

print("\nFinal cleaned dataset:")
print(df.head())