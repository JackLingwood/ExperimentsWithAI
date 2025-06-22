import matplotlib.pyplot as plt

# Sample data
x = [1, 2, 3, 4, 5]
y = [10, 12, 8, 14, 7]

# Create a new figure
plt.figure()

# Plot the data
plt.plot(x, y, label="Sales Over Time", marker='o')

# Add a title and labels
plt.title("Simple Line Chart Example")
plt.xlabel("Day")
plt.ylabel("Sales")

# Add a legend
plt.legend()

# Show a grid
plt.grid(True)

# Display the plot
plt.gcf().canvas.manager.set_window_title('Simple Line Chart Example')
plt.tight_layout()
plt.show()
