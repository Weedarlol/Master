import matplotlib.pyplot as plt
import numpy as np
import os
import re


# FETCHING DATA FROM FILES
#____________________________________________________________________________________
folder_path_2d_ex3 = "../ex3/CPU_2d/output"

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

    sorted_list = sorted(info_list, key=lambda x: (x[0], x[1]))
    
    return sorted_list

info_list = process_files(folder_path_2d_ex3)
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



# Creates area where we want to land
# Constants
iterations = 10000
bandwidth_dgx2q = 23.84 * 2**30
bandwidth_hgx2q = 23.84 * 2**30
bandwidth_fox = 23.84 * 2**30
bandwidth = 23.84 * 2**30



def plot_info(grouped_info_list):
    # Separate the elements based on the partition
    partitions = {}
    matrix_sizes = set()
    for string, elements in grouped_info_list:
        for element in elements:
            partition = element[3]
            matrix_size = f"{element[0]}x{element[1]}"
            matrix_sizes.add(matrix_size)
            if partition not in partitions:
                partitions[partition] = {'x_values': [], 'y_values': []}
            partitions[partition]['x_values'].append(matrix_size)
            partitions[partition]['y_values'].append(element[4])

        

    print(partitions)

    for partition, data in partitions.items():
        plt.plot(data['x_values'], data['y_values'], label=partition)

    # Set title and labels
    plt.title(f"Iterations: {info_list[0][2]}")
    plt.xlabel("Matrix Size")
    plt.ylabel("Time (s)")

    # Rotate x-axis labels for better visibility
    plt.xticks(rotation=45)

    # Add legend
    plt.legend()

    # Show the plot
    plt.tight_layout()
    plt.show()

# Plot the info_list
plot_info(grouped_info_list)