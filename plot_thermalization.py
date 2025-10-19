import matplotlib.pyplot as plt
import numpy as np


input_file = './plaquette_thermalization_570_6x6_1.dat'



print(f"Loading data from: {input_file}")


try:
    data = np.loadtxt(input_file, comments='#')
except FileNotFoundError:
    print(f"Error: The file '{input_file}' was not found.")
    print("Please make sure you have run 'plaq_v_cfg.py' first and the filename is correct.")
    exit()

# Separate the data into columns
config_number = data[:, 0]
plaquette_value = data[:, 1]

print("Data loaded successfully. Generating plot...")

# Create the plot
plt.figure(figsize=(12, 6))
plt.plot(config_number, plaquette_value, marker='.', linestyle='-', markersize=2)

# Add labels and a title
plt.xlabel('Configuration Number (Monte Carlo Time)')
plt.ylabel('Average Plaquette')
plt.title('Plaquette Thermalization History')
plt.grid(True)

# Save the plot to a file
output_image = 'thermalization_plot.png'
plt.savefig(output_image, dpi=300)
print(f"Plot saved to: {output_image}")

# Display the plot on the screen
plt.show()