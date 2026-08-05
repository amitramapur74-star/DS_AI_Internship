print("Welcome to Shopping")

cart = []

while True:
    item = input("Enter your item: ")
    cart.append(item)

    choice = input("Do you want to add another item? (yes/no): ")

    if choice == "no":
        break

print("\nDone")

print("Cart type:", type(cart))
print("Total items:", len(cart))
print("Cart:", cart)

cart = tuple(cart)

print("\nCart type:", type(cart))
print("Checkout:", cart)