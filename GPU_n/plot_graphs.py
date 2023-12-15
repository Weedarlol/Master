import os
import re
import matplotlib.pyplot as plt
import numpy as np

def extract_info(file_path):
    # Extracting data from filename
    filename = os.path.basename(file_path)
    match = re.match(r'.*width(\d+)_height(\d+)_gpu(\d+)_iter(\d+)_compare(\d+)_overlap(\d+)_.*', filename)
    if match:
        width = match.group(1)
        height = match.group(2)
        gpu = match.group(3)
        iterationsComputed = match.group(4)
        overlap = match.group(6)

    # Extracting data from file content
    with open(file_path, 'r') as file:
        content = file.read()
        time_match = re.search(r'Time - (\d+\.\d+)', content)
        if time_match:
            time = time_match.group(1)

    return width, height, gpu, iterationsComputed, overlap, float(time)

# Directory containing the files
directory = 'output'  # Replace 'path_to_directory' with your actual directory path

# Initialize dictionaries to store data for different width, height, and iterationsComputed combinations
data_by_dimensions = {}

# List all files in the directory
files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

# Extract information from files
for file in files:
    file_path = os.path.join(directory, file)
    if file.endswith('.out'):
        width, height, gpu, iterationsComputed, overlap, time = extract_info(file_path)
        key = (width, height, iterationsComputed)
        if key not in data_by_dimensions:
            data_by_dimensions[key] = {'gpu': [], 'overlap_0': [], 'overlap_1': []}
        if overlap == '0':
            data_by_dimensions[key]['gpu'].append(int(gpu))
            data_by_dimensions[key]['overlap_0'].append(time)
        elif overlap == '1':
            data_by_dimensions[key]['overlap_1'].append(time)

# Create separate bar plots for different width, height, and iterationsComputed combinations
for key, data in data_by_dimensions.items():
    width, height, iterationsComputed = key
    x = np.array(data['gpu'])
    y_overlap_0 = np.array(data['overlap_0'])
    y_overlap_1 = np.array(data['overlap_1'])

    # Sort based on GPU number
    sort_order = np.argsort(x)
    x = x[sort_order]
    y_overlap_0 = y_overlap_0[sort_order]
    y_overlap_1 = y_overlap_1[sort_order]

    bar_width = 0.35  # Width of the bars
    index = np.arange(len(x))  # Index for the x-axis

    plt.figure()
    plt.bar(index, y_overlap_0, bar_width, label='Overlap = False')
    plt.bar(index + bar_width, y_overlap_1, bar_width, label='Overlap = True')

    plt.xlabel('GPU Quantity')
    plt.ylabel('Time(s)')
    plt.title(f'Bar Plot for width {width}, height {height}, iterationsComputed {iterationsComputed}')
    plt.xticks(index + bar_width / 2, x)
    plt.legend()
    plt.tight_layout()
    plt.show()
