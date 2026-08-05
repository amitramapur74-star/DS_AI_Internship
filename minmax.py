x = [32,44,63,22]

n = int(input("Enter how many numbers: "))

for i in range(n):
    num = int(input("Enter number: "))
    x.append(num)

print("\nList:", x)
print("Minimum:", min(x))
print("Maximum:", max(x))
print("Sum:", sum(x))
print("Average:", sum(x) / len(x))
print("Total Length:", len(x))
print("Sorted List:", sorted(x))