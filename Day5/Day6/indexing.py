import pandas as pd

# Create a Pandas Series with subject names as labels
marks = pd.Series(
    [75, 55, 82, 68, 45],
    index=["Maths", "Python", "DBMS", "Data Science", "Networks"]
)

print("Student Marks:")
print(marks)

# Access value using position
print("\nValue at position 0:", marks.iloc[0])

# Access value using label
print("Marks in Python:", marks["Python"])

# Print values
print("\nValues:")
print(marks.values)

# Print index
print("\nIndex:")
print(marks.index)

# Boolean masking - students/subjects scoring above 60
print("\nMarks above 60:")
print(marks[marks > 60])
