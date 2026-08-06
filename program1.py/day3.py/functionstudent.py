def student_result():
    name = input("Enter student name: ")

    marks = []
    n = int(input("Enter number of subjects: "))

    for i in range(n):
        mark = int(input(f"Enter mark {i+1}: "))
        marks.append(mark)

    avg = sum(marks) / len(marks)

    if avg >= 90:
        grade = "A+"
    elif avg >= 80:
        grade = "A"
    elif avg >= 70:
        grade = "B"
    elif avg >= 60:
        grade = "C"
    elif avg >= 50:
        grade = "D"
    else:
        grade = "F"

    print("\n----- Student Report -----")
    print("Name :", name)
    print("Marks :", marks)
    print("Average :", avg)
    print("Grade :", grade)

student_result()