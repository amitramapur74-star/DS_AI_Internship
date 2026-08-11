import pandas as pd

# Create a Pandas Series of student marks
marks = pd.Series(
    [75, 55, 82, 68, 45],
    index=["Maths", "Physics", "Chemistry", "Python", "English"]
)

# Print the Series
print("Student Marks:")
print(marks)

# Access value using position
print("\nMark at position 0:", marks.iloc[0])

# Access value using label
print("Mark in Maths:", marks["Maths"])

# Print values
print("\nValues:")
print(marks.values)

# Print index
print("\nIndex:")
print(marks.index)

# Boolean masking: students/subjects with marks above 60
print("\nMarks above 60:")
print(marks[marks > 60])