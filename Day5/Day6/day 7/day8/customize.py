import matplotlib.pyplot as plt
plt.subplot(1, 2, 1)
plt.plot([1, 2, 3], [4, 5, 6])  
plt.title("Line Plot")
plt.subplot(1, 2, 2)
plt.bar(['A', 'B', 'C'], [10, 20, 15])
plt.title("Bar Chart")  
plt.show()