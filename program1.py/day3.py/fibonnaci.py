def square_table():
    n = int(input("Enter number of terms: "))

    for i in range(1, n + 1):
        print(f"{i} * {i} = {i * i}")

square_table()