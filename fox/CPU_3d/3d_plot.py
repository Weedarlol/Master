import matplotlib.pyplot as plt
import numpy as np
import os
import re

folder_path = "output/"

def extract_info_from_file(filename):
    # Define the pattern for extracting information from the filename
    pattern = r"output/width(\d+)_height(\d+)_depth(\d+)_iter(\d+)_createGrid(\d+)_(\w+)\.out"
    # Match the pattern in the filename
    match_name = re.match(pattern, filename)


    with open(filename, 'r') as file:
        content = file.read()
    match_content = re.search(r"Time\(event\) - (\d+\.\d+) s", content)
    
    if match_name and match_content:
        # Extract information from the matched groups
        width, height, depth, iterations, create_matrix, y_string = match_name.groups()
        width, height, depth, iterations, create_matrix = map(int, (width, height, depth, iterations, create_matrix))
        time = float(match_content.group(1))
        return width, height, depth, iterations, y_string, time
    else:
        return None

def process_files(folder_path):
    # Get a list of all files in the specified folder
    files = os.listdir(folder_path)

    info_list = []
    
    for file_name in files:
        # Construct the full path to the file
        file_path = os.path.join(folder_path, file_name)

        info = extract_info_from_file(file_path)


        if info:
            info_list.append(info)

    sorted_list = sorted(info_list, key=lambda x: (x[0], x[1], x[2]))
    
    return sorted_list

info_list = process_files(folder_path)

# Constants
iterations = 10000
bandwidth = 23.84 * 2**30

# Extract information from the sorted list
widths = [info[0] for info in info_list]
heights = [info[1] for info in info_list]
depths = [info[2] for info in info_list]
elements = [a * b * c for a, b, c in zip(widths, heights, depths)]
memory_operations = [2, 3, 4, 5, 6, 7]
y_values_1 = [(memory_operations[0] * w * h * d * 8 * iterations) / bandwidth for w, h, d in zip(widths, heights, depths)]
y_values_2 = [(memory_operations[1] * w * h * d* 8 * iterations) / bandwidth for w, h, d in zip(widths, heights, depths)]
y_values_3 = [(memory_operations[2] * w * h * d* 8 * iterations) / bandwidth for w, h, d in zip(widths, heights, depths)]
y_values_4 = [(memory_operations[3] * w * h * d* 8 * iterations) / bandwidth for w, h, d in zip(widths, heights, depths)]
y_values_5 = [(memory_operations[4] * w * h * d* 8 * iterations) / bandwidth for w, h, d in zip(widths, heights, depths)]
y_values_6 = [(memory_operations[5] * w * h * d* 8 * iterations) / bandwidth for w, h, d in zip(widths, heights, depths)]
y_values_time = [info[5] for info in info_list]

# Create plot with twin y-axis
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

# Plot the first two lines on the left y-axis
line1 = ax1.plot(elements, y_values_1, label='Memory Operations = 2', color='blue', marker='o')
line2 = ax1.plot(elements, y_values_2, label='Memory Operations = 3', color='lightblue', marker='o')
line3 = ax1.plot(elements, y_values_3, label='Memory Operations = 4', color='lightblue', marker='v')
line4 = ax1.plot(elements, y_values_4, label='Memory Operations = 5', color='lightblue', marker='v')
line5 = ax1.plot(elements, y_values_5, label='Memory Operations = 6', color='lightblue', marker='s')
line6 = ax1.plot(elements, y_values_6, label='Memory Operations = 7', color='blue', marker='s')

# Plot the third line on the right y-axis
line7 = ax1.plot(elements, y_values_time, label='Computational Time', color='red', marker='^')
area = ax1.fill_between(elements, y_values_1, y_values_6, color='lightgray', alpha=0.5)

# Set labels and title for the left y-axis
ax1.set_xlabel('Bytes')
ax1.set_ylabel('Time (s) - Memory Operations', color='black')
ax1.tick_params(axis='y', labelcolor='black')

# Set labels and title for the right y-axis
ax2.set_ylabel('percentage execution time vs 5 memory operations', color='black')
ax2.tick_params(axis='y', labelcolor='black')

# Add legend for all lines
lines = line1 + line2 + line3 + line4 + line5 + line6 + line7
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, loc='upper left')

# Set logarithmic scale for x-axis
ax1.set_xscale('log', base=2)



# Calculate percentage difference and plot as horizontal line
percentage_difference = [((t - m) / m) * 100 for t, m in zip(y_values_time, y_values_4)]


line6 = ax2.plot(elements, percentage_difference, color='black', linestyle='--', marker='o', label='Percentage difference')

# Add legend for twinx axis
ax2.legend(loc='upper right')
ax2.set_ylim(bottom=0)

# Show the plot
plt.show()
