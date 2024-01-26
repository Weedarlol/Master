import os
import re
import matplotlib.pyplot as plt
import numpy as np

def extract_info(file_path):
    # Extracting data from filename
    filename = os.path.basename(file_path)
    match = re.match(r'.*width(\d+)_height(\d+)_gpu(\d+)_iter(\d+)_compare(\d+)_overlap(\d+)_.*', filename)
    if match:
        width = int(match.group(1))
        height = int(match.group(2))
        gpu = int(match.group(3))
        iterationsComputed = int(match.group(4))
        overlap = int(match.group(6))

    # Extracting data from file content
    with open(file_path, 'r') as file:
        content = file.read()
        time_match = re.search(r'Time\(event\) - (\d+\.\d+)', content)
        if time_match:
            time = float(time_match.group(1))

    return width, height, gpu, iterationsComputed, overlap, time

# Initialize dictionaries to store data for different width, height, and iterationsComputed combinations
data_by_dimensions = {}

# List all files in the directory
files = [f for f in os.listdir('output') if os.path.isfile(os.path.join('output', f))]

# Extract information from files
for file in files:
    file_path = os.path.join('output', file)
    if file.endswith('.out'):
        width, height, gpu, iterationsComputed, overlap, time = extract_info(file_path)
        key = (width, height, iterationsComputed)
        if key not in data_by_dimensions:
            data_by_dimensions[key] = {'gpu': [], 'overlap_0': [], 'overlap_1': [], 'overlap_2': []}
        if overlap == 0:
            data_by_dimensions[key]['gpu'].append(gpu)
            data_by_dimensions[key]['overlap_0'].append(time)
        elif overlap == 1:
            data_by_dimensions[key]['overlap_1'].append(time)
        elif overlap == 2:
            data_by_dimensions[key]['overlap_2'].append(time)

# Get unique width and height values
unique_widths = sorted(set(key[0] for key in data_by_dimensions))
unique_heights = sorted(set(key[1] for key in data_by_dimensions))

# Create subplots layout based on unique width and height values
num_rows = len(unique_heights)
num_cols = len(unique_widths)

# Create separate bar plots for different width, height, and iterationsComputed combinations
fig, axes = plt.subplots(num_rows, num_cols, sharex='col', sharey='row', figsize=(15, 10))

for i, height in enumerate(unique_heights):
    for j, width in enumerate(unique_widths):
        key = (width, height, iterationsComputed)
        data = data_by_dimensions.get(key, {'gpu': [], 'overlap_0': [], 'overlap_1': [], 'overlap_2': []})

        x = np.array(data['gpu'])
        y_overlap_0 = np.array(data['overlap_0'])
        y_overlap_1 = np.array(data['overlap_1'])
        y_overlap_2 = np.array(data['overlap_2'])

        # Sort based on GPU number
        sort_order = np.argsort(x)
        x = x[sort_order]
        y_overlap_0 = y_overlap_0[sort_order]
        y_overlap_1 = y_overlap_1[sort_order]
        y_overlap_2 = y_overlap_2[sort_order]

        bar_width = 0.25  # Width of the bars
        index = np.arange(len(x))  # Index for the x-axis

        # Plot on the corresponding subplot with explicit colors
        axes[i, j].bar(index - bar_width, y_overlap_0, bar_width, label='Overlap = 0', color='blue')
        axes[i, j].bar(index, y_overlap_1, bar_width, label='Overlap = 1', color='orange')
        axes[i, j].bar(index + bar_width, y_overlap_2, bar_width, label='Overlap = 2', color='green')

        # Set subplot labels
        axes[i, j].set_xlabel('GPU Quantity')
        axes[i, j].set_ylabel('Time(ms)')
        axes[i, j].set_title(f'width {width}, height {height}, iterationsComputed {iterationsComputed}')

        # Set x-axis ticks to show only GPU quantity values
        axes[i, j].set_xticks(index)
        axes[i, j].set_xticklabels(x)

        # Add legend to each subplot
        axes[i, j].legend()

# Extract partition name from the first file title (assuming all files have the same partition name)
first_file = files[0]
partition_match = re.search(r'.*overlap(\d+)_(\w+)', first_file)
partition_name = partition_match.group(2) if partition_match else 'Unknown Partition'

# Set common labels and main title with the partition name
fig.suptitle(f'Partition: {partition_name}', fontsize=16)

# Show all x-axis and y-axis values
for ax in axes.flat:
    ax.tick_params(axis='both', which='both', labelbottom=True, labelleft=True)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the layout to include the title
plt.show()
