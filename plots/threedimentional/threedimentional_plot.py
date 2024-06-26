import matplotlib.pyplot as plt
import numpy as np
import math
import os
import re

# FETCHING DATA FROM FILES
#____________________________________________________________________________________

def extract_info_from_file(filename):
    # Define the pattern for extracting information from the filename
    pattern = r".*/width(\d+)_height(\d+)_depth(\d+)_gpu(\d+)_iter(\d+)_compare(\d+)_overlap(\d+)_test(\d+)_createGrid(\d+)_(\w+)\.out"
    # Match the pattern in the filename
    match_name = re.match(pattern, filename)
    with open(filename, 'r') as file:
        content = file.read()
    match_content = re.search(r"Time\(event\) - (\d+\.\d+) s", content)
    if match_name and match_content:
        # Extract information from the matched groups
        width, height, depth, gpu, iterations, compare, overlap, test, create_grid, y_string = match_name.groups()
        width, height, depth, gpu, iterations, compare, overlap, test, create_grid = map(int, (width, height, depth, gpu, iterations, compare, overlap, test, create_grid))
        time = float(match_content.group(1))
        return width, height, depth, gpu, iterations, overlap, test, y_string, time
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

    sorted_list = sorted(info_list, key=lambda x: (x[0], x[1], x[2], x[3]))
    
    return sorted_list


#_____________________________________________________________________________________________
def group_by_string(info_list):
    # Create a dictionary to hold the grouped elements
    grouped_info = {}
    # Iterate over the elements in info_list
    for info in info_list:
        # Extract the string from the info tuple
        string = info[7]
        # If the string is not already in the dictionary, create a new list for it
        if string not in grouped_info:
            grouped_info[string] = []
        # Append the info tuple to the list corresponding to the string
        grouped_info[string].append(info)
    # Convert the dictionary to a list of tuples
    grouped_info_list = [(string, elements) for string, elements in grouped_info.items()]
    return grouped_info_list

def plot_info_cpu(grouped_info_list, save_option):
    grouped_info_list = [
        (partition, [(x, y, z, w, u, v, a, b, c) for x, y, z, w, u, v, a, b, c in elements if v == 0])
        for partition, elements in grouped_info_list
    ]

    num_partitions = len(grouped_info_list) + 1
    num_rows = math.ceil(math.sqrt(num_partitions))
    num_cols = math.ceil(num_partitions / num_rows)
    _, axes = plt.subplots(num_rows, num_cols, figsize=(10, 8), sharex=True, sharey=True)
    bandwidth_values = [
        ("dgx2q", 11.107 * 2**30),
        ("hgx2q", 26.625 * 2**30),
        ("accel", 21.861 * 2**30)
    ]
    for i, (partition, elements) in enumerate(grouped_info_list):
        row_idx = i // num_cols
        col_idx = i % num_cols

        x_values = [f"{element[0]}x{element[1]}x{element[2]}" for element in elements]
        y_values = [element[-1] for element in elements]
    
        for string, value in bandwidth_values:
            if partition == string:
                # Extract information from the sorted list
                memory_operations = [2, 3, 4, 5, 6, 7]
                y_values_1 = [(memory_operations[0] * x * y * z * 8 * 10000) / value for x, y, z in [(element[0], element[1], element[2]) for element in elements]]
                y_values_2 = [(memory_operations[1] * x * y * z * 8 * 10000) / value for x, y, z in [(element[0], element[1], element[2]) for element in elements]]
                y_values_3 = [(memory_operations[2] * x * y * z * 8 * 10000) / value for x, y, z in [(element[0], element[1], element[2]) for element in elements]]
                y_values_4 = [(memory_operations[3] * x * y * z * 8 * 10000) / value for x, y, z in [(element[0], element[1], element[2]) for element in elements]]
                y_values_5 = [(memory_operations[4] * x * y * z * 8 * 10000) / value for x, y, z in [(element[0], element[1], element[2]) for element in elements]]
                y_values_6 = [(memory_operations[5] * x * y * z * 8 * 10000) / value for x, y, z in [(element[0], element[1], element[2]) for element in elements]]
                # Plot the first two lines on the left y-axis
                axes[row_idx, col_idx].plot(x_values, y_values_1, label='Memory Operations = 2', color='blue', marker='o')
                axes[row_idx, col_idx].plot(x_values, y_values_2, label='Memory Operations = 3', color='lightblue', marker='o')
                axes[row_idx, col_idx].plot(x_values, y_values_3, label='Memory Operations = 4', color='lightgreen', marker='o')
                axes[row_idx, col_idx].plot(x_values, y_values_4, label='Memory Operations = 5', color='lightgreen', marker='s')
                axes[row_idx, col_idx].plot(x_values, y_values_5, label='Memory Operations = 6', color='lightblue', marker='s')
                axes[row_idx, col_idx].plot(x_values, y_values_6, label='Memory Operations = 7', color='blue', marker='s')
                axes[row_idx, col_idx].fill_between(x_values, y_values_1, y_values_6, color='lightgray', alpha=0.5)

        axes[row_idx, col_idx].plot(x_values, y_values, label=partition, color="red", marker='x')
        axes[row_idx, col_idx].set_title(f"Partition: {elements[0][7]}")
        axes[row_idx, col_idx].set_ylabel("Time (s)")
        axes[row_idx, col_idx].legend()

        axes[-1, -1].plot(x_values, y_values, label=partition, marker='x')
        axes[-1, -1].set_title(f"Combined")
        axes[-1, -1].set_ylabel("Time (s)")
        axes[-1, -1].legend()

    for ax in plt.gcf().get_axes():
        ax.grid(axis='y')  # Add gridlines along the y-axis for each subplot
        ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.suptitle("Estimated CPU Computation Time vs Real CPU Computation Time")
    plt.subplots_adjust(top=0.9)
    if save_option == "yes":
        plt.savefig('output_figures/3d_CPU_Computation_Time.png')
    plt.show()

def plot_info_cpu_cpu(grouped_info_list, save_option):
    grouped_info_list = [
        (partition, [(x, y, z, w, u, v, a, b, c) for x, y, z, w, u, v, a, b, c in elements if v == 0])
        for partition, elements in grouped_info_list
    ]

    num_partitions = len(grouped_info_list) + 1
    num_rows = math.ceil(math.sqrt(num_partitions))
    num_cols = math.ceil(num_partitions / num_rows)
    _, axes = plt.subplots(num_rows, num_cols, figsize=(10, 8), sharex=True, sharey=True)
    
    

def plot_info_gpu(grouped_info_list, save_option):
    grouped_info_list = [
        (partition, [(x, y, z, w, u, v, a, b, c) for x, y, z, w, u, v, a, b, c in elements if a == 0 and v == 1])
        for partition, elements in grouped_info_list
    ]

    num_partitions = len(grouped_info_list)
    num_rows = math.ceil(math.sqrt(num_partitions))
    num_cols = math.ceil(num_partitions / num_rows)

    _, axes = plt.subplots(num_rows, num_cols, figsize=(10, 8), sharex=True, sharey=True)
    for i, (partition, elements) in enumerate(grouped_info_list):
        row_idx = i // num_cols
        col_idx = i % num_cols

        gpu_num = [f"{element[3]}" for element in elements]
        gpus = dict.fromkeys(gpu_num)
        gpus = [int(key) for key in gpus.keys()]
        gpu_num = len(gpus)

        x_values = [f"{element[0]}x{element[1]}x{element[2]}" for element in elements]
        x_values = list(dict.fromkeys(x_values))
        y_values = [[] for _ in range(gpu_num)]
        for element in elements:
            y_values[element[3]-1].append(element[-1])
        

        for j, gpu_number in enumerate(list(gpus)):
            if len(y_values[j]) < len(x_values):
                x_values_tmp = x_values[:len(y_values[j])]
            else:
                x_values_tmp = x_values

            if(num_cols > 1):
                axes[row_idx, col_idx].plot(x_values_tmp, y_values[j], label=f'Number of GPUs = {gpu_number}', marker='o')
                
                axes[row_idx, col_idx].set_title(f"Partition: {partition}")
                axes[row_idx, col_idx].set_ylabel("Time (s)")
                axes[row_idx, col_idx].legend()
            else:
                axes[row_idx].plot(x_values_tmp, y_values[j], label=f'Number of GPUs = {gpu_number}', marker='o')
                axes[row_idx].set_title(f"Partition: {partition}")
                axes[row_idx].set_ylabel("Time (s)")
                axes[row_idx].legend()
            
    for ax in plt.gcf().get_axes():
        ax.grid(axis='y')  # Add gridlines along the y-axis for each subplot
        ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.suptitle("GPU Computation time, Overlap versus No Overlap")
    plt.subplots_adjust(top=0.9)
    if save_option == "yes":
        plt.savefig('output_figures/3d_CPU_Computation_Time.png')
    plt.show()

def plot_overlap_gpu(grouped_info_list, save_option):
    grouped_info_list = [
        (partition, [(x, y, z, w, u, v, a, b, c) for x, y, z, w, u, v, a, b, c in elements if a == 0 and x >= 512])
        for partition, elements in grouped_info_list
    ]

    num_rows = 3
    num_cols = len(grouped_info_list)
    _, axes = plt.subplots(num_rows, num_cols, figsize=(10, 8))

    for i, (partition, elements) in enumerate(grouped_info_list):
        x_values = [f"{element[0]}x{element[1]}x{element[2]}" for element in elements]
        x_values = list(dict.fromkeys(x_values))
        y_values = [[] for _ in range(num_rows*num_cols*2)]
        for element in elements:
            if element[5] == 0:
                y_values[element[3]*2-4].append(element[8])
            else:
                y_values[element[3]*2-3].append(element[8])
        

        percentage_diff = []
        for j in range(0, len(y_values), 2):
            sublist1 = y_values[j]
            sublist2 = y_values[j + 1]
            diff = [(a - b) / b * 100 for a, b in zip(sublist1, sublist2)]
            percentage_diff.append(diff)


        if(num_cols > 1):
            for row in range(num_rows):
                if len(y_values[row*2]) < len(x_values):
                    x_values = x_values[:len(y_values[row*2])]
                elif len(y_values[row*2+1]) < len(x_values):
                    x_values = x_values[:len(y_values[row*2+1])]
                axes[row, i].plot(x_values, y_values[row*2], label='No Overlap')
                axes[row, i].plot(x_values, y_values[row*2+1], label='Overlap')
                axes[row, i].legend()
                axes[row, i].set_ylabel("Time (s)")
                ax2 = axes[row, i].twinx()
                ax2.plot(x_values, percentage_diff[row][:len(x_values)], marker='^', color='g', linestyle='-', label='Percentage Difference')
                ax2.legend(loc='upper right')
                ax2.set_ylabel("Percentage Difference")
                
        else:
            for row in range(num_rows):
                if len(y_values[row*2]) < len(x_values):
                    x_values = x_values[:len(y_values[row*2])]
                elif len(y_values[row*2+1]) < len(x_values):
                    x_values = x_values[:len(y_values[row*2+1])]
                axes[row].plot(x_values, y_values[row*2], label='No Overlap')
                axes[row].plot(x_values, y_values[row*2+1], label='Overlap')
                axes[row].set_title(f"{partition} - GPUs: {row+2}")
                axes[row].legend(loc='upper left')
                axes[row].set_ylabel("Time (s)")
                ax2 = axes[row].twinx()
                ax2.plot(x_values, percentage_diff[row][:len(x_values)], marker='^', color='g', linestyle='-', label='Percentage Difference')
                ax2.legend(loc='upper right')
                ax2.set_ylabel("Percentage Difference")
        
    for ax in plt.gcf().get_axes():
        ax.grid(axis='y')  # Add gridlines along the y-axis for each subplot
        ax.tick_params(axis='x', rotation=45)
    plt.xlabel("Matrix Size")
    plt.suptitle("GPU Computation time, Overlap versus No Overlap")
    plt.subplots_adjust(top=0.9)
    plt.tight_layout()  
    if save_option == "yes":
        plt.savefig('output_figures/3d_GPU_Overlap_vs_No_Overlap.png')
    plt.show()

def plot_estimate_gpu(grouped_info_list, save_option):
    grouped_info_list = [
        (partition, [(x, y, z, w, u, v, a-2 if a > 0 else a, b, c) for x, y, z, w, u, v, a, b, c in elements])
        for partition, elements in grouped_info_list
    ]

    num_rows = 3
    num_cols = len(grouped_info_list)
    _, axes = plt.subplots(num_rows, num_cols, figsize=(10, 8), sharex=True, sharey=False)

    for i, (partition, elements) in enumerate(grouped_info_list):
        x_values = [f"{element[0]}x{element[1]}x{element[2]}" for element in elements]
        x_values = list(dict.fromkeys(x_values))

        y_value_overlap = [[] for _ in range(num_rows*3)]
        y_value_no_overlap = [[] for _ in range(num_rows*3)]
        
        for element in elements:
            if element[5] == 1:
                y_value_overlap[(element[3] - 2) * 3 + element[6]].append(element[8])
            else:
                y_value_no_overlap[(element[3] - 2) * 3 + element[6]].append(element[8])

        if(num_cols > 1):
            for row in range(num_rows):
                if len(y_value_overlap[row*3]) < len(x_values):
                    x_values = x_values[:len(y_value_overlap[row*3])]
                elif len(y_value_overlap[row*3+1]) < len(x_values):
                    x_values = x_values[:len(y_value_overlap[row*3+1])]
                axes[row, i].plot(x_values, y_value_overlap[row*3], label='Overlap', color="red")
                axes[row, i].plot(x_values, y_value_no_overlap[row*3], label='No Overlap', color="blue")
                axes[row, i].fill_between(x_values, [x + y for x, y in zip(y_value_overlap[row*3+2], y_value_overlap[row*3+1])], color='moccasin', label="Communication + Computation")
                axes[row, i].fill_between(x_values, y_value_overlap[row*3+1], color='gray', label='Only Computation')
                axes[row, i].set_title(f"{partition} - GPUs: {row+2}")
                axes[row, i].legend()
                axes[row, i].set_ylabel("Time (s)")
        else:
            for row in range(num_rows):
                if len(y_value_overlap[row*2]) < len(x_values):
                    x_values = x_values[:len(y_value_overlap[row*2])]
                elif len(y_value_overlap[row*2+1]) < len(x_values):
                    x_values = x_values[:len(y_value_overlap[row*2+1])]
                axes[row].plot(x_values, y_value_overlap[row], label='Full time')
                axes[row].plot(x_values, y_value_overlap[row + 1*num_rows], label='Only Computation')
                axes[row].plot(x_values, y_value_overlap[row + 2*num_rows], label='Only Communication')
                axes[row].set_title(f"{partition} - GPUs: {row+2}")
                axes[row].legend(loc='upper left')
                axes[row].set_ylabel("Time (s)")

    for ax in plt.gcf().get_axes():
        ax.grid(axis='y')  # Add gridlines along the y-axis for each subplot
        ax.tick_params(axis='x', rotation=45)
    plt.xlabel("Matrix Size")
    plt.suptitle("GPU computation time difference: Total, Computation and Communication. Overlap")
    plt.subplots_adjust(top=0.9)
    plt.tight_layout()  
    if save_option == "yes":
        plt.savefig('output_figures/3d_GPU_Total_Computation_Communication.png')
    plt.show()


    _, axes = plt.subplots(num_rows, num_cols, figsize=(10, 8), sharex=True, sharey=False)

    for i, (partition, elements) in enumerate(grouped_info_list):
        x_values = [f"{element[0]}x{element[1]}x{element[2]}" for element in elements]
        x_values = list(dict.fromkeys(x_values))

        y_value_overlap = [[] for _ in range(num_rows*3)]
        y_value_no_overlap = [[] for _ in range(num_rows*3)]
        
        for element in elements:
            if element[5] == 1:
                y_value_overlap[(element[3] - 2) * 3 + element[6]].append(element[8])
            else:
                y_value_no_overlap[(element[3] - 2) * 3 + element[6]].append(element[8])

        if(num_cols > 1):
            for row in range(num_rows):
                if len(y_value_overlap[row*3]) < len(x_values):
                    x_values = x_values[:len(y_value_overlap[row*3])]
                elif len(y_value_overlap[row*3+1]) < len(x_values):
                    x_values = x_values[:len(y_value_overlap[row*3+1])]
                axes[row, i].plot(x_values, y_value_overlap[row*3], label='Overlap', color="red")
                axes[row, i].plot(x_values, y_value_no_overlap[row*3], label='No Overlap', color="blue")
                axes[row, i].fill_between(x_values, [x + y for x, y in zip(y_value_no_overlap[row*3+2], y_value_no_overlap[row*3+1])], color='moccasin', label="Communication + Computation")
                axes[row, i].fill_between(x_values, y_value_no_overlap[row*3+1], color='gray', label='Only Computation')
                axes[row, i].set_title(f"{partition} - GPUs: {row+2}")
                axes[row, i].legend()
                axes[row, i].set_ylabel("Time (s)")
        else:
            for row in range(num_rows):
                if len(y_value_overlap[row*2]) < len(x_values):
                    x_values = x_values[:len(y_value_overlap[row*2])]
                elif len(y_value_overlap[row*2+1]) < len(x_values):
                    x_values = x_values[:len(y_value_overlap[row*2+1])]
                axes[row].plot(x_values, y_value_overlap[row], label='Full time')
                axes[row].plot(x_values, y_value_overlap[row + 1*num_rows], label='Only Computation')
                axes[row].plot(x_values, y_value_overlap[row + 2*num_rows], label='Only Communication')
                axes[row].set_title(f"{partition} - GPUs: {row+2}")
                axes[row].legend(loc='upper left')
                axes[row].set_ylabel("Time (s)")

    for ax in plt.gcf().get_axes():
        ax.grid(axis='y')  # Add gridlines along the y-axis for each subplot
        ax.tick_params(axis='x', rotation=45)
    plt.xlabel("Matrix Size")
    plt.suptitle("GPU computation time difference: Total, Computation and Communication. No overlap")
    plt.subplots_adjust(top=0.9)
    plt.tight_layout()  
    if save_option == "yes":
        plt.savefig('output_figures/3d_GPU_Total_Computation_Communication.png')
    plt.show()   

def plot_bandwidth_gpu(grouped_info_list, save_option):
    grouped_info_list = [
        (partition, [(x, y, z, w, u, v, a, b, c) for x, y, z, w, u, v, a, b, c in elements if v == 1 and a ==  0])
        for partition, elements in grouped_info_list
    ]
    num_rows = 3
    num_cols = len(grouped_info_list)
    bandwidth_values = [
                    ("dgx2q", 900 * 2**30),
                    ("hgx2q", 900 * 2**30),
                    ("accel", 900 * 2**30)
                ]

    _, axes = plt.subplots(num_rows, num_cols, figsize=(10, 8), sharex=True, sharey=True)
    for i, (partition, elements) in enumerate(grouped_info_list):
        x_values = [f"{element[0]}x{element[1]}x{element[2]}" for element in elements]
        x_values = list(dict.fromkeys(x_values))
        y_values = [element[-1] for element in elements]
    
        for j in range(num_rows):
            new_y_values = [y_values[i+j] for i in range(0, len(y_values), num_rows)]
            for string, value in bandwidth_values:
                if partition == string:
                    # Extract information from the sorted list
                    memory_operations = [2, 3, 4, 5, 6, 7]
                    unique_combinations = sorted(set((element[0], element[1], element[2]) for element in elements), key=lambda x: (x[0], x[1], x[2]))
                    y_values_1 = [(memory_operations[0] * x * y * z * 8 * 10000) / value for x, y, z in unique_combinations]
                    y_values_2 = [(memory_operations[1] * x * y * z * 8 * 10000) / value for x, y, z in unique_combinations]
                    y_values_3 = [(memory_operations[2] * x * y * z * 8 * 10000) / value for x, y, z in unique_combinations]
                    y_values_4 = [(memory_operations[3] * x * y * z * 8 * 10000) / value for x, y, z in unique_combinations]
                    y_values_5 = [(memory_operations[4] * x * y * z * 8 * 10000) / value for x, y, z in unique_combinations]
                    y_values_6 = [(memory_operations[5] * x * y * z * 8 * 10000) / value for x, y, z in unique_combinations]
                    # Plot the first two lines on the left y-axis
                    if(num_cols > 1):
                        axes[j, i].plot(x_values, y_values_1, label='Memory Operations = 2', color='blue', marker='o')
                        axes[j, i].plot(x_values, y_values_2, label='Memory Operations = 3', color='lightblue', marker='o')
                        axes[j, i].plot(x_values, y_values_3, label='Memory Operations = 4', color='lightgreen', marker='o')
                        axes[j, i].plot(x_values, y_values_4, label='Memory Operations = 5', color='lightgreen', marker='s')
                        axes[j, i].plot(x_values, y_values_5, label='Memory Operations = 6', color='lightblue', marker='s')
                        axes[j, i].plot(x_values, y_values_6, label='Memory Operations = 7', color='blue', marker='s')
                        axes[j, i].fill_between(x_values, y_values_1, y_values_6, color='lightgray', alpha=0.5)
                    else:
                        axes[j].plot(x_values, y_values_1, label='Memory Operations = 2', color='blue', marker='o')
                        axes[j].plot(x_values, y_values_2, label='Memory Operations = 3', color='lightblue', marker='o')
                        axes[j].plot(x_values, y_values_3, label='Memory Operations = 4', color='lightgreen', marker='o')
                        axes[j].plot(x_values, y_values_4, label='Memory Operations = 5', color='lightgreen', marker='s')
                        axes[j].plot(x_values, y_values_5, label='Memory Operations = 6', color='lightblue', marker='s')
                        axes[j].plot(x_values, y_values_6, label='Memory Operations = 7', color='blue', marker='s')
                        axes[j].fill_between(x_values, y_values_1, y_values_6, color='lightgray', alpha=0.5)
            if num_cols > 1:
                print(x_values)
                print(y_values)
                axes[j, i].plot(x_values, y_values, label=partition, color="red", marker='x')
                axes[j, i].set_title(f"Partition: {partition}")
                axes[j, i].set_ylabel("Time (s)")
                axes[j, i].legend()
            else:
                axes[j].plot(x_values, y_values, label=partition, color="red", marker='x')
                axes[j].set_title(f"Partition: dgx2q")
                axes[j].set_ylabel("Time (s)")
                axes[j].legend()
    # Rotate x-axis labels for better visibility
    for ax in plt.gcf().get_axes():
        ax.grid(axis='y')  # Add gridlines along the y-axis for each subplot
        ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.suptitle("Estimated GPU Computation Time vs Real GPU Computation Time")
    if save_option == "yes":
        plt.savefig('output_figures/3d_GPU_Estimated_vs_Real.png')
    plt.show()
