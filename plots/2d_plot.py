import matplotlib.pyplot as plt
import numpy as np
import math
import os
import re

folder_path_2d_ex3_cpu = "../ex3/CPU_2d/output"
folder_path_2d_fox_cpu = "../fox/CPU_2d/output"
folder_path_2d_ex3_gpu = "../ex3/GPU_2d/output"
folder_path_2d_fox_gpu = "../fox/GPU_2d/output"

# FETCHING DATA FROM FILES
#____________________________________________________________________________________

def extract_info_from_file(filename):
    # Define the pattern for extracting information from the filename
    pattern = r".*/width(\d+)_height(\d+)_gpu(\d+)_iter(\d+)_compare(\d+)_overlap(\d+)_test(\d+)_createMatrix(\d+)_(\w+)\.out"
    # Match the pattern in the filename
    match_name = re.match(pattern, filename)
    with open(filename, 'r') as file:
        content = file.read()
    match_content = re.search(r"Time\(event\) - (\d+\.\d+) s", content)
    if match_name and match_content:
        # Extract information from the matched groups
        width, height, gpu, iterations, compare, overlap, test, create_matrix, y_string = match_name.groups()
        width, height, gpu, iterations, compare, overlap, test, create_matrix = map(int, (width, height, gpu, iterations, compare, overlap, test, create_matrix))
        time = float(match_content.group(1))
        return width, height, gpu, iterations, overlap, test, y_string, time
    else:
        return None


def process_files(folder_path, existing_info_list=None):
    # Get a list of all files in the specified folder
    files = os.listdir(folder_path)

    info_list = existing_info_list if existing_info_list else []
    
    for file_name in files:
        # Construct the full path to the file
        file_path = os.path.join(folder_path, file_name)

        info = extract_info_from_file(file_path)

        if info:
            info_list.append(info)

    sorted_list = sorted(info_list, key=lambda x: (x[0], x[1]))
    
    return sorted_list


#_____________________________________________________________________________________________
def group_by_string(info_list):
    # Create a dictionary to hold the grouped elements
    grouped_info = {}
    # Iterate over the elements in info_list
    for info in info_list:
        # Extract the string from the info tuple
        string = info[6]
        # If the string is not already in the dictionary, create a new list for it
        if string not in grouped_info:
            grouped_info[string] = []
        # Append the info tuple to the list corresponding to the string
        grouped_info[string].append(info)
    # Convert the dictionary to a list of tuples
    grouped_info_list = [(string, elements) for string, elements in grouped_info.items()]
    return grouped_info_list


def plot_info_cpu(grouped_info_list):
    num_partitions = len(grouped_info_list)
    num_rows = math.ceil(math.sqrt(num_partitions))
    num_cols = math.ceil(num_partitions / num_rows)
    bandwidth_values = [
        ("dgx2q", 19.87 * 2**30),
        ("hgx2q", 23.84 * 2**30)
    ]
    _, axes = plt.subplots(num_rows, num_cols, figsize=(10, 8), sharex=True, sharey=True)
    for i, (partition, elements) in enumerate(grouped_info_list):
        row_idx = i // num_cols
        col_idx = i % num_cols
        # Prepare data for plotting
        x_values = [f"{element[0]}x{element[1]}" for element in elements]
        y_values = [element[7] for element in elements]
        size = [int(size.split('x')[0]) for size in x_values]
        for string, value in bandwidth_values:
            if partition == string:
                # Extract information from the sorted list
                memory_operations = [2, 3, 4, 5]
                y_values_1 = [(memory_operations[0] * s * s * 8 * 10000) / value for s in size]
                y_values_2 = [(memory_operations[1] * s * s * 8 * 10000) / value for s in size]
                y_values_3 = [(memory_operations[2] * s * s * 8 * 10000) / value for s in size]
                y_values_4 = [(memory_operations[3] * s * s * 8 * 10000) / value for s in size]
                # Plot the first two lines on the left y-axis
                axes[row_idx, col_idx].plot(x_values, y_values_1, label='Memory Operations = 2', color='blue', marker='o')
                axes[row_idx, col_idx].plot(x_values, y_values_2, label='Memory Operations = 3', color='lightblue', marker='o')
                axes[row_idx, col_idx].plot(x_values, y_values_3, label='Memory Operations = 4', color='lightblue', marker='s')
                axes[row_idx, col_idx].plot(x_values, y_values_4, label='Memory Operations = 5', color='blue', marker='s')
                axes[row_idx, col_idx].fill_between(x_values, y_values_1, y_values_4, color='lightgray', alpha=0.5)
        axes[row_idx, col_idx].plot(x_values, y_values, label=partition, color="red", marker='x')
        axes[row_idx, col_idx].set_title(f"Partition: {elements[0][6]}")
        axes[row_idx, col_idx].set_ylabel("Time (s)")
        axes[row_idx, col_idx].legend()
    # Plot all lines in the last subplot
    for i, (partition, elements) in enumerate(grouped_info_list):
        x_values = [f"{element[0]}x{element[1]}" for element in elements]
        y_values = [element[7] for element in elements]
        axes[-1, -1].plot(x_values, y_values, label=partition, marker='x')
        axes[-1, -1].set_title(f"Combined")
        axes[-1, -1].set_ylabel("Time (s)")
        axes[-1, -1].legend()
    # Rotate x-axis labels for better visibility
    for ax in plt.gcf().get_axes():
        ax.grid(axis='y')  # Add gridlines along the y-axis for each subplot
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()


def plot_overlap_gpu(grouped_info_list):
    sor_filtered_list = [
        (partition, sorted([(x, y, z, w, u, v, a, b) for x, y, z, w, u, v, a, b in elements if v == 0], key=lambda x: x[2]))
        for partition, elements in grouped_info_list
    ]
    grouped_info_list = [
        (partition, sorted([(x, y, z, w, u, v, a, b) for x, y, z, w, u, v, a, b in elements], key=lambda x: x[4]))
        for partition, elements in sor_filtered_list
    ]

    unique_strings = set()
    unique_values = set()
    unique_x_values = set()
    
    for _, elements in grouped_info_list:
        for element in elements:
            unique_strings.add(element[6])
            unique_values.add(element[2])
            unique_x_values.add(element[0])  # Collect unique x values

    num_rows = len(unique_values)
    num_cols = len(unique_strings)

    _, axes = plt.subplots(num_rows, num_cols, figsize=(10, 8), sharex=True, sharey=False)

    for i, (partition, elements) in enumerate(grouped_info_list):
        # Initialize dictionaries to store x and y values for each subplot
        x_values_dict = {}
        y_values_dict = {}
        
        for element in elements:
            string_idx = list(unique_strings).index(element[6])
            value_idx = list(unique_values).index(element[2])

            x_label = str(element[0])
            y_value = element[-1]

            # Group x and y values by subplot index
            if (value_idx, string_idx) not in x_values_dict:
                x_values_dict[(value_idx, string_idx)] = []
                y_values_dict[(value_idx, string_idx)] = []
                
            x_values_dict[(value_idx, string_idx)].append(x_label)
            y_values_dict[(value_idx, string_idx)].append(y_value)
        
        # Plot each group of x and y values as lines on the corresponding subplot
        for (value_idx, string_idx), (x_values, y_values) in zip(x_values_dict.keys(), zip(x_values_dict.values(), y_values_dict.values())):
            # Split x_values and y_values into two parts at the halfway point
            half_index = len(x_values) // 2
            x_values_1, x_values_2 = x_values[:half_index], x_values[half_index:]
            y_values_1, y_values_2 = y_values[:half_index], y_values[half_index:]
            
            # Plot first line
            axes[value_idx, string_idx].plot(x_values_1, y_values_1, marker='o', color='b', linestyle='-', label='No Overlap')
            # Plot second line
            axes[value_idx, string_idx].plot(x_values_2, y_values_2, marker='s', color='r', linestyle='--', label='Overlap')

            # Calculate percentage difference between the two lines
            percentage_diff = [(y1 - y2) / y1 * 100 for y1, y2 in zip(y_values_1, y_values_2)]
            
            # Create secondary y-axis
            ax2 = axes[value_idx, string_idx].twinx()
            # Plot percentage difference line on secondary y-axis
            ax2.plot(x_values[half_index:], percentage_diff, marker='^', color='g', linestyle='-', label='Percentage Difference')

            # Add grid
            axes[value_idx, string_idx].grid(True)

    for i, string in enumerate(unique_strings):
        for j, value in enumerate(unique_values):
            axes[j, i].set_title(f"{string} - GPUs: {value}")
            axes[j, i].tick_params(axis='x', rotation=45)
            axes[j, i].legend()
            ax2.legend(loc='upper right')
            

    # Set common labels and title
    plt.xlabel("Matrix Size")
    plt.ylabel("Time (s)")
    plt.suptitle("Overlap by GPU Type")
    plt.tight_layout()
    plt.show()















def plot_estimate_gpu(grouped_info_list):
    sort_filtered_list = [
        (partition, sorted([(x, y, z, w, u, v, a, b) for x, y, z, w, u, v, a, b in elements if u == 1], key=lambda x: x[2]))
        for partition, elements in grouped_info_list
    ]
    grouped_info_list = [
        (partition, sorted([(x, y, z, w, u, v, a, b) for x, y, z, w, u, v, a, b in elements if a == "dgx2q"], key=lambda x: x[4]))
        for partition, elements in sort_filtered_list
    ]

    unique_strings = set()
    unique_values = set()
    unique_x_values = set()
    unique_test = set()
    
    for _, elements in grouped_info_list:
        for element in elements:
            unique_strings.add(element[6])
            unique_values.add(element[2])
            unique_x_values.add(element[0])
            unique_test.add(element[5])

    num_rows = len(unique_values)
    num_cols = len(unique_strings)

    _, axes = plt.subplots(num_rows, num_cols, figsize=(10, 8), sharex=True, sharey=False)

    for i, (partition, elements) in enumerate(grouped_info_list):
        x_values_dict = {}
        y_values_dict = {}
        row_idx = i // num_cols
        col_idx = i % num_cols
        
        for element in elements:
            string_idx = list(unique_strings).index(element[6])
            value_idx = list(unique_values).index(element[2])

            x_label = str(element[0])
            y_value = element[-1]

            # Group x and y values by subplot index
            if (value_idx, string_idx) not in x_values_dict:
                x_values_dict[(value_idx, string_idx)] = []
                y_values_dict[(value_idx, string_idx)] = []
                
            x_values_dict[(value_idx, string_idx)].append(x_label)
            y_values_dict[(value_idx, string_idx)].append(y_value)

        for (value_idx, string_idx), (x_values, y_values) in zip(x_values_dict.keys(), zip(x_values_dict.values(), y_values_dict.values())):
            y_data = [y_values[i::len(unique_test)] for i in range(len(unique_test))]

        for i, data in enumerate(y_data):
            axes[row_idx, col_idx].plot(x_values[::len(unique_test)], y_data, marker='o', label=f'Line {i+1}')

        """ for i, y_data_line in enumerate(y_data):
            axes[row_idx, col_idx].plot(x_values[::len(y_data_line)], y_data_line, marker='o', label=f'Line {i+1}')
 """
    # Set common labels and title
    plt.xlabel("Matrix Size")
    plt.ylabel("Time (s)")
    plt.suptitle("Overlap by GPU Type")
    plt.tight_layout()
    plt.show()

















info_list_cpu = []
info_list_cpu = process_files(folder_path_2d_ex3_cpu, info_list_cpu)
info_list_cpu = process_files(folder_path_2d_fox_cpu, info_list_cpu)
info_list_gpu = []
info_list_gpu = process_files(folder_path_2d_ex3_gpu, info_list_gpu)


# Call the function to group the info_list by the string
#grouped_info_list_cpu = group_by_string(info_list_cpu)
grouped_info_list_gpu = group_by_string(info_list_gpu)

#filtered_list = [(gpu_type, [elem for elem in elements if elem[-3] == 0]) for gpu_type, elements in grouped_info_list_gpu]



# Plot the info_list
#plot_overlap_gpu(grouped_info_list_gpu)
plot_estimate_gpu(grouped_info_list_gpu)
