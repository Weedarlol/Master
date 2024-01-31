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
def extract_numerical_values(filename):
    match = re.match(r'.*width(\d+)_height(\d+)_gpu(\d+)_iter(\d+)_compare(\d+)_overlap(\d+)_.*', filename)
    if match:
        return [int(match.group(i)) for i in range(1, 7)]
    return []

# Directory containing the files
directory = 'output'
unsorted_files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

# Custom sorting function
def custom_sort(filename):
    numerical_values = extract_numerical_values(filename)
    return numerical_values

# Sort the list of files using the custom sorting function
files = sorted(unsorted_files, key=custom_sort)

# Extract information from files
for file in files:
    file_path = os.path.join('output', file)
    if file.endswith('.out'):
        width, height, gpu, iterationsComputed, overlap, time = extract_info(file_path)
        key = (width, height, iterationsComputed)
        if key not in data_by_dimensions:
            data_by_dimensions[key] = {'gpu': [], 'overlap_0': [], 'overlap_1': []}
        if overlap == 0:
            data_by_dimensions[key]['gpu'].append(gpu)
            data_by_dimensions[key]['overlap_0'].append(time)
        elif overlap == 1:
            data_by_dimensions[key]['overlap_1'].append(time)

# Get unique width and height values
unique_widths = sorted(set(key[0] for key in data_by_dimensions))
unique_heights = sorted(set(key[1] for key in data_by_dimensions))

# Create separate bar plots for different width, height, and iterationsComputed combinations
fig, axes = plt.subplots(len(unique_heights), len(unique_widths), sharex='col', figsize=(15, 10))

for i, height in enumerate(unique_heights):
    for j, width in enumerate(unique_widths):
        key = (width, height, iterationsComputed)  # Assuming iterationsComputed is not relevant for subplot arrangement
        data = data_by_dimensions.get(key, {'gpu': [], 'overlap_0': [], 'overlap_1': []})

        x = np.array(data['gpu'])
        y_overlap_0 = np.array(data['overlap_0'])
        y_overlap_1 = np.array(data['overlap_1'])

        # Sort based on GPU number
        sort_order = np.argsort(x)
        x_sorted = x[sort_order]
        y_overlap_0_sorted = y_overlap_0[sort_order]
        y_overlap_1_sorted = y_overlap_1[sort_order]

        bar_width = 0.35  # Width of the bars
        index = np.arange(len(x_sorted))  # Index for the x-axis

        # Plot on the corresponding subplot with explicit colors
        bars_overlap_0 = axes[i, j].bar(index, y_overlap_0_sorted, bar_width, label='Overlap = False', color='blue')
        bars_overlap_1 = axes[i, j].bar(index + bar_width, y_overlap_1_sorted, bar_width, label='Overlap = True', color='orange')

        # Set subplot labels
        axes[i, j].set_xlabel('GPU Quantity')
        axes[i, j].set_ylabel('Time(ms)')
        axes[i, j].set_title(f'width {width}, height {height}, iterationsComputed {iterationsComputed}')

        # Set x-axis ticks and labels
        axes[i, j].set_xticks(index + bar_width / 2)
        axes[i, j].set_xticklabels(x_sorted)

        # Add legend to each subplot
        axes[i, j].legend()

        # Add text labels on top of each bar
        for bar_, x_val, y_val in zip([bars_overlap_0, bars_overlap_1], x_sorted, [y_overlap_0_sorted, y_overlap_1_sorted]):
            for bar_container in bar_:
                height_print = bar_container.get_height()
                axes[i, j].text(
                    bar_container.get_x() + bar_container.get_width() / 2, height_print, f'{height_print:.2f}', ha='center', va='bottom'
                )

fig, ax_improvement = plt.subplots(figsize=(10, 6))

for i, height in enumerate(unique_heights):
    for j, width in enumerate(unique_widths):
        key = (width, height, iterationsComputed)
        data = data_by_dimensions.get(key, {'gpu': [], 'overlap_0': [], 'overlap_1': []})

        x_sorted = np.array(data['gpu'])
        y_overlap_0_sorted = np.array(data['overlap_0'])[sort_order]
        y_overlap_1_sorted = np.array(data['overlap_1'])[sort_order]

        # Calculate percentage improvement for each GPU quantity
        percentage_improvement = (y_overlap_0_sorted - y_overlap_1_sorted) / y_overlap_0_sorted * 100

        line_width = 2  # Width of the line plot
        index = np.arange(len(x_sorted))  # Index for the x-axis

        # Plot the percentage improvement on the same plot
        ax_improvement.plot(x_sorted, percentage_improvement, label=f'width {width}, height {height}', marker='o')
        ax_improvement.set_xticks(x_sorted);

# Set plot labels
ax_improvement.set_xlabel('GPU Quantity')
ax_improvement.set_ylabel('Percentage Improvement (%)')
ax_improvement.set_title(f'Percentage Improvement for Different Width and Height')
ax_improvement.legend()

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
