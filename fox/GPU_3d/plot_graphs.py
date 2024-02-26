import os
import re
import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text

def extract_info(file_path):
    # Extracting data from filename
    filename = os.path.basename(file_path)
    match = re.match(r'.*width(\d+)_height(\d+)_gpu(\d+)_iter(\d+)_compare(\d+)_overlap(\d+)_test(\d+)_.*', filename)
    if match:
        width = int(match.group(1))
        height = int(match.group(2))
        gpu = int(match.group(3))
        iterationsComputed = int(match.group(4))
        overlap = int(match.group(6))
        test = int(match.group(7))

    # Extracting data from file content
    with open(file_path, 'r') as file:
        content = file.read()
        time_match = re.search(r'Time\(event\) - (\d+\.\d+)', content)
        if time_match:
            time = float(time_match.group(1))

    return width, height, gpu, iterationsComputed, overlap, time, test

# Initialize dictionaries to store data for different width, height, and iterationsComputed combinations
data_by_dimensions = {}

# List all files in the directory
def extract_numerical_values(filename):
    match = re.match(r'.*width(\d+)_height(\d+)_gpu(\d+)_iter(\d+)_compare(\d+)_overlap(\d+)_test(\d+)_.*', filename)
    if match:
        return [int(match.group(i)) for i in range(1, 8)]
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

# Extract partition names and test integers from the files
partition_test_combinations = set()
for file in files:
    partition_match = re.search(r'.*test(\d+)_(\w+)', file)
    test_partition_name = (int(partition_match.group(1)), partition_match.group(2)) if partition_match else (0, 'Unknown Partition')
    partition_test_combinations.add(test_partition_name)

# Plot data for each partition and test combination
for test, partition_name in sorted(partition_test_combinations):
    # Initialize dictionaries to store data for different width, height, and iterationsComputed combinations
    data_by_dimensions = {}

    # Extract files for the current partition and test
    partition_test_files = [file for file in files if f'_test{test}_{partition_name}.' in file]

    if(test == 0):
        test_name = "Full Calculation"
    elif(test == 1):
        test_name = "No Kernel"
    else:
        test_name = "No Communication"

    # Extract information from files
    for file in partition_test_files:
        file_path = os.path.join('output', file)
        width, height, gpu, iterationsComputed, overlap, time, current_test = extract_info(file_path)
        key = (width, height, iterationsComputed)
        if key not in data_by_dimensions:
            data_by_dimensions[key] = {'gpu': [], 'overlap_0': [], 'overlap_1': []}
        if overlap == 0:
            data_by_dimensions[key]['gpu'].append(gpu)
            data_by_dimensions[key]['overlap_0'].append(time)
        elif overlap == 1:
            data_by_dimensions[key]['overlap_1'].append(time)

    # Get unique width and height values for the current partition
    unique_widths_partition = sorted(set(key[0] for key in data_by_dimensions))
    unique_heights_partition = sorted(set(key[1] for key in data_by_dimensions))

    fig, axes = plt.subplots(len(unique_heights_partition), len(unique_widths_partition), sharex='col', figsize=(15, 10))
    fig.subplots_adjust(wspace=0.2, hspace=0.3)

    legend_handles = []
    legend_labels_added = set()
    # Set common labels and main title with the partition name and test number
    fig.suptitle(f'Partition: {partition_name}, Test: {test_name}', fontsize=16)

    # Plot data for the current partition and test on the corresponding subplots
    for i, height in enumerate(unique_heights_partition):
        for j, width in enumerate(unique_widths_partition):
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

            handles_overlap_0, labels_overlap_0 = axes[i, j].get_legend_handles_labels()
            for handle, label in zip(handles_overlap_0, labels_overlap_0):
                if label not in legend_labels_added:
                    legend_handles.append(handle)
                    legend_labels_added.add(label)

            for bar_, x_val, y_val in zip([bars_overlap_0, bars_overlap_1], x_sorted, [y_overlap_0_sorted, y_overlap_1_sorted]):
                for bar_container in bar_:
                    height_print = bar_container.get_height()
                    axes[i, j].text(
                        bar_container.get_x() + bar_container.get_width() / 2, height_print, f'{height_print:.2f}', ha='center', va='bottom'
                    )

    legend = fig.legend(legend_handles, legend_labels_added, loc='upper center', bbox_to_anchor=(0.5, 0.94), ncol=len(unique_widths_partition))
    legend.get_texts()[0]
    legend.get_texts()[1]
    # Set the font size for the legend text
    for text in legend.get_texts():
        text.set_fontsize(14)  # You can adjust the font size as needed

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 

    fig, ax_improvement = plt.subplots(figsize=(10, 6))
    texts = []

    for i, height in enumerate(unique_heights_partition):
        for j, width in enumerate(unique_widths_partition):
            key = (width, height, iterationsComputed)
            data = data_by_dimensions.get(key, {'gpu': [], 'overlap_0': [], 'overlap_1': []})

            x = np.array(data['gpu'])
            y_overlap_0_sorted = np.array(data['overlap_0'])
            y_overlap_1_sorted = np.array(data['overlap_1'])

            # Calculate percentage improvement for each GPU quantity
            percentage_improvement = (y_overlap_0_sorted - y_overlap_1_sorted) / y_overlap_0_sorted * 100

            line_width = 2  # Width of the line plot
            index = np.arange(len(x_sorted))  # Index for the x-axis

            # Plot the percentage improvement on the same plot
            if len(x_sorted) == len(percentage_improvement):
                ax_improvement.plot(x_sorted, percentage_improvement, label=f'width {width}, height {height}', marker='o', color=plt.cm.hsv(((i**2)+(j**2))/((len(unique_heights_partition)**2+len(unique_widths_partition)**2)/2)))
                text_1 = ax_improvement.text(x_sorted[0], percentage_improvement[0], f'width {width}, height {height}')
                text_2 = ax_improvement.text(x_sorted[-1], percentage_improvement[-1], f'width {width}, height {height}')
            elif len(x) == len(percentage_improvement):
                ax_improvement.plot(x, percentage_improvement, label=f'width {width}, height {height}', marker='o', color=plt.cm.hsv(((i**2)+(j**2))/((len(unique_heights_partition)**2+len(unique_widths_partition)**2)/2)))
                text_1 = ax_improvement.text(x_sorted[0], percentage_improvement[0], f'width {width}, height {height}')
                text_2 = ax_improvement.text(x_sorted[-1], percentage_improvement[-1], f'width {width}, height {height}')
           
            ax_improvement.set_xticks(x_sorted)

            texts.append(text_1)
            texts.append(text_2)

    ax_improvement.set_xlabel('GPU Quantity')
    ax_improvement.set_ylabel('Percentage Improvement (%)')
    ax_improvement.set_title(f'Percentage Improvement for Different Width and Height')
    ax_improvement.axhline(y=0, color='black', linestyle='--', linewidth=1)
    adjust_text(texts, arrowprops=dict(arrowstyle="->", color='r', lw=0.5), only_move='x-,y-')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the layout to include the title

plt.show()
