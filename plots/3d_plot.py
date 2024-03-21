import matplotlib.pyplot as plt
import numpy as np
import math
import os
import re


# FETCHING DATA FROM FILES
#____________________________________________________________________________________
folder_path_3d_ex3 = "../ex3/CPU_3d/output"
folder_path_3d_fox = "../fox/CPU_3d/output"

def extract_info_from_file(filename):
    # Define the pattern for extracting information from the filename
    pattern = r".*/width(\d+)_height(\d+)_iter(\d+)_createMatrix(\d+)_(\w+)\.out"
    # Match the pattern in the filename
    match_name = re.match(pattern, filename)

    with open(filename, 'r') as file:
        content = file.read()
    match_content = re.search(r"Time\(event\) - (\d+\.\d+) s", content)
    
    if match_name and match_content:
        # Extract information from the matched groups
        width, height, iterations, create_matrix, y_string = match_name.groups()
        width, height, iterations, create_matrix = map(int, (width, height, iterations, create_matrix))
        time = float(match_content.group(1))
        return width, height, iterations, y_string, time
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

info_list = []
info_list = process_files(folder_path_2d_ex3, info_list)
info_list = process_files(folder_path_2d_fox, info_list)


#_____________________________________________________________________________________________
def group_by_string(info_list):
    # Create a dictionary to hold the grouped elements
    grouped_info = {}

    # Iterate over the elements in info_list
    for info in info_list:
        # Extract the string from the info tuple
        string = info[3]
        # If the string is not already in the dictionary, create a new list for it
        if string not in grouped_info:
            grouped_info[string] = []
        # Append the info tuple to the list corresponding to the string
        grouped_info[string].append(info)

    # Convert the dictionary to a list of tuples
    grouped_info_list = [(string, elements) for string, elements in grouped_info.items()]

    return grouped_info_list

# Call the function to group the info_list by the string
grouped_info_list = group_by_string(info_list)


def plot_info(grouped_info_list):
    num_partitions = len(grouped_info_list)
    num_rows = math.ceil(math.sqrt(num_partitions))
    num_cols = math.ceil(num_partitions / num_rows)


    bandwidth_values = [
        ("dgx2q", 19.87 * 2**30),
        ("hgx2q", 23.84 * 2**30)
    ]

    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 8), sharex=True, sharey=True)

    for i, (partition, elements) in enumerate(grouped_info_list):
        row_idx = i // num_cols
        col_idx = i % num_cols

        # Prepare data for plotting
        x_values = [f"{element[0]}x{element[1]}" for element in elements]
        y_values = [element[4] for element in elements]

        size = [int(size.split('x')[0]) for size in x_values]

        for string, value in bandwidth_values:
            if partition == string:
                # Extract information from the sorted list
                widths = [info[0] for info in info_list]
                heights = [info[1] for info in info_list]
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



        axes[row_idx, col_idx].plot(x_values, y_values, label=partition, color="red")

        axes[row_idx, col_idx].set_title(f"Iterations: {elements[0][2]}")
        axes[row_idx, col_idx].set_ylabel("Time (s)")
        axes[row_idx, col_idx].legend()

    # Plot all lines in the last subplot
    for i, (partition, elements) in enumerate(grouped_info_list):
        x_values = [f"{element[0]}x{element[1]}" for element in elements]
        y_values = [element[4] for element in elements]
        
        axes[-1, -1].plot(x_values, y_values, label=partition)

        axes[row_idx, col_idx].set_title(f"Iterations: {elements[0][2]}")
        axes[row_idx, col_idx].set_ylabel("Time (s)")
        axes[row_idx, col_idx].legend()


    # Rotate x-axis labels for better visibility
    for ax in plt.gcf().get_axes():
        ax.grid(axis='y')  # Add gridlines along the y-axis for each subplot
        ax.tick_params(axis='x', rotation=45)
 

    plt.tight_layout()
    plt.show()

# Plot the info_list
plot_info(grouped_info_list)