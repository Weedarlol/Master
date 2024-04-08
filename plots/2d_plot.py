import matplotlib.pyplot as plt
import numpy as np
import math
import os
import re

def extract_info_from_file(filename):
    pattern = r".*/width(\d+)_height(\d+)_gpu(\d+)_iter(\d+)_compare(\d+)_overlap(\d+)_test(\d+)_createMatrix(\d+)_(\w+)\.out"
    match_name = re.match(pattern, filename)
    with open(filename, 'r') as file:
        content = file.read()
    match_content = re.search(r"Time\(event\) - (\d+\.\d+) s", content)
    if match_name and match_content:
        width, height, gpu, iterations, compare, overlap, test, create_matrix, y_string = match_name.groups()
        width, height, gpu, iterations, compare, overlap, test, create_matrix = map(int, (width, height, gpu, iterations, compare, overlap, test, create_matrix))
        time = float(match_content.group(1))
        return width, height, gpu, iterations, overlap, test, y_string, time
    else:
        return None


def process_files(folder_path, existing_info_list=None):
    files = os.listdir(folder_path)
    info_list = existing_info_list if existing_info_list else []
    for file_name in files:
        file_path = os.path.join(folder_path, file_name)
        info = extract_info_from_file(file_path)
        if info:
            info_list.append(info)
    sorted_list = sorted(info_list, key=lambda x: (x[0], x[1], x[2]))
    return sorted_list


def group_by_string(info_list):
    grouped_info = {}
    for info in info_list:
        string = info[6]
        if string not in grouped_info:
            grouped_info[string] = []
        grouped_info[string].append(info)
    grouped_info_list = [(string, elements) for string, elements in grouped_info.items()]
    return grouped_info_list


def plot_info_cpu(grouped_info_list):
    grouped_info_list = [
        (partition, [(x, y, z, w, u, v, a, b) for x, y, z, w, u, v, a, b in elements if v == 0])
        for partition, elements in grouped_info_list
    ]

    num_partitions = len(grouped_info_list) + 1
    num_rows = math.ceil(math.sqrt(num_partitions))
    num_cols = math.ceil(num_partitions / num_rows)
    _, axes = plt.subplots(num_rows, num_cols, figsize=(10, 8), sharex=True, sharey=True)
    bandwidth_values = [
        ("dgx2q", 11.107 * 2**30),
        ("hgx2q", 26.625 * 2**30),
        ("accel", 19.0205 * 2**30)
    ]
    for i, (partition, elements) in enumerate(grouped_info_list):
        row_idx = i // num_cols
        col_idx = i % num_cols

        x_values = [f"{element[0]}x{element[1]}" for element in elements]
        y_values = [element[-1] for element in elements]
    
        for string, value in bandwidth_values:
            if partition == string:
                # Extract information from the sorted list
                memory_operations = [2, 3, 4, 5]
                y_values_1 = [(memory_operations[0] * s * s * 8 * 10000) / value for s in [element[0] for element in elements]]
                y_values_2 = [(memory_operations[1] * s * s * 8 * 10000) / value for s in [element[0] for element in elements]]
                y_values_3 = [(memory_operations[2] * s * s * 8 * 10000) / value for s in [element[0] for element in elements]]
                y_values_4 = [(memory_operations[3] * s * s * 8 * 10000) / value for s in [element[0] for element in elements]]
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
    plt.show()



def plot_overlap_gpu(grouped_info_list):
    grouped_info_list = [
        (partition, [(x, y, z, w, u, v, a, b) for x, y, z, w, u, v, a, b in elements if v == 0])
        for partition, elements in grouped_info_list
    ]

    num_rows = 3
    num_cols = len(grouped_info_list)
    _, axes = plt.subplots(num_rows, num_cols, figsize=(10, 8))

    for i, (partition, elements) in enumerate(grouped_info_list):
        x_values = [f"{element[0]}x{element[1]}" for element in elements]
        x_values = list(dict.fromkeys(x_values))
        y_values = [[] for _ in range(num_rows*num_cols*2)]

        for element in elements:
            if element[4] == 0:
                y_values[element[2]*2-4].append(element[7])
            else:
                y_values[element[2]*2-3].append(element[7])
        
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
                axes[row, i].set_title(f"{partition} - GPUs: {row+2}")
                axes[row, i].legend(loc='upper left')
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
        
      
    plt.xlabel("Matrix Size")
    plt.suptitle("GPU Computation time, Overlap versus No Overlap")
    plt.subplots_adjust(top=0.9)
    plt.tight_layout()  
    plt.show()



def plot_estimate_gpu(grouped_info_list):
    grouped_info_list = [
        (partition, [(x, y, z, w, u, v-2 if v > 0 else v, a, b) for x, y, z, w, u, v, a, b in elements if u == 1 and x == y])
        for partition, elements in grouped_info_list
    ]

    num_rows = 3
    num_cols = len(grouped_info_list)
    _, axes = plt.subplots(num_rows, num_cols, figsize=(10, 8), sharex=True)

    for i, (partition, elements) in enumerate(grouped_info_list):
        x_values = [f"{element[0]}x{element[1]}" for element in elements]
        x_values = list(dict.fromkeys(x_values))

        y_values = [[] for _ in range(num_rows*num_cols*3)]

        for element in elements:
            y_values[(element[2] - 2)*3 + element[5]].append(element[7])

        if(num_cols > 1):
            for row in range(num_rows):
                if len(y_values[row*3]) < len(x_values):
                    x_values = x_values[:len(y_values[row*3])]
                elif len(y_values[row*3+1]) < len(x_values):
                    x_values = x_values[:len(y_values[row*3+1])]
                axes[row, i].plot(x_values, y_values[row*3], label='Full time')
                axes[row, i].plot(x_values, y_values[row*3+1], label='Only Computation')
                axes[row, i].plot(x_values, y_values[row*3+2], label='Only Communication')
                axes[row, i].set_title(f"{partition} - GPUs: {row+2}")
                axes[row, i].legend()
                axes[row, i].set_ylabel("Time (s)")
        else:
            for row in range(num_rows):
                if len(y_values[row*2]) < len(x_values):
                    x_values = x_values[:len(y_values[row*2])]
                elif len(y_values[row*2+1]) < len(x_values):
                    x_values = x_values[:len(y_values[row*2+1])]
                axes[row].plot(x_values, y_values[row], label='Full time')
                axes[row].plot(x_values, y_values[row + 1*num_rows], label='Only Computation')
                axes[row].plot(x_values, y_values[row + 2*num_rows], label='Only Communication')
                axes[row].set_title(f"{partition} - GPUs: {row+2}")
                axes[row].legend(loc='upper left')
                axes[row].set_ylabel("Time (s)")

    plt.xlabel("Matrix Size")
    plt.suptitle("GPU computation time difference: Total, Computation and Communication")
    plt.subplots_adjust(top=0.9)
    plt.tight_layout()  
    plt.show()


def plot_bandwidth_gpu(grouped_info_list):
    grouped_info_list = [
        (partition, [(x, y, z, w, u, v, a, b) for x, y, z, w, u, v, a, b in elements if u == 0 and v == 0 and x == y])
        for partition, elements in grouped_info_list
    ]
    num_rows = 1
    num_cols = len(grouped_info_list)
    bandwidth_values = [
        ("dgx2q", 871.29 * 2**30),
        ("hgx2q", 1702.88 * 2**30)
    ]

    print(num_cols)

    _, axes = plt.subplots(num_rows, num_cols, figsize=(10, 8), sharex=True, sharey=True)
    for i, (partition, elements) in enumerate(grouped_info_list):
        x_values = [f"{element[0]}x{element[1]}" for element in elements]
        x_values = list(dict.fromkeys(x_values))
        first_integers = [int(x.split('x')[0]) for x in x_values]
        second_integers = [int(x.split('x')[1]) for x in x_values]
        y_values = [element[-1] for element in elements]

        for string, value in bandwidth_values:
            if partition == string:
                # Extract information from the sorted list
                memory_operations = [2, 3, 4, 5]
                y_values_1 = [(memory_operations[0] * a * b * 8 * 10000) / value for a, b in zip(first_integers, second_integers)]
                y_values_2 = [(memory_operations[1] * a * b * 8 * 10000) / value for a, b in zip(first_integers, second_integers)]
                y_values_3 = [(memory_operations[2] * a * b * 8 * 10000) / value for a, b in zip(first_integers, second_integers)]
                y_values_4 = [(memory_operations[3] * a * b * 8 * 10000) / value for a, b in zip(first_integers, second_integers)]
                axes[i].plot(x_values, y_values_1, label='Memory Operations = 2', color='blue', marker='o')
                axes[i].plot(x_values, y_values_2, label='Memory Operations = 3', color='lightblue', marker='o')
                axes[i].plot(x_values, y_values_3, label='Memory Operations = 4', color='lightblue', marker='s')
                axes[i].plot(x_values, y_values_4, label='Memory Operations = 5', color='blue', marker='s')
                axes[i].fill_between(x_values, y_values_1, y_values_4, color='lightgray', alpha=0.5)

        axes[i].plot(x_values, y_values, label=partition, color="red", marker='x')
        axes[i].set_title(f"Partition: {partition}")
        axes[i].set_ylabel("Time (s)")
        axes[i].legend()

    # Rotate x-axis labels for better visibility
    for ax in plt.gcf().get_axes():
        ax.grid(axis='y')  # Add gridlines along the y-axis for each subplot
        ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.suptitle("Estimated GPU Computation Time vs Real GPU Computation Time")
    plt.show()










folder_path_ex3_cpu = "../ex3/CPU_2d/output"
folder_path_fox_cpu = "../fox/CPU_2d/output"

folder_path_ex3_gpu = "../ex3/GPU_2d/output"
folder_path_fox_gpu = "../fox/GPU_2d/output"

folder_path_ex3_1gpu = "../ex3/GPU_2d_1GPU/output"
folder_path_fox_1gpu = "../fox/GPU_2d_1GPU/output"

info_list_cpu = []
info_list_cpu = process_files(folder_path_ex3_cpu, info_list_cpu)
info_list_cpu = process_files(folder_path_fox_cpu, info_list_cpu)
info_list_gpu = []
info_list_gpu = process_files(folder_path_ex3_gpu, info_list_gpu)
info_list_1gpu = []
info_list_1gpu = process_files(folder_path_ex3_1gpu, info_list_1gpu)
info_list_1gpu = process_files(folder_path_fox_1gpu, info_list_1gpu)



# Call the function to group the info_list by the string
grouped_info_list_cpu = group_by_string(info_list_cpu)
grouped_info_list_gpu = group_by_string(info_list_gpu)
grouped_info_list_1gpu = group_by_string(info_list_1gpu)


#plot_info_cpu(grouped_info_list_cpu)
#plot_overlap_gpu(grouped_info_list_gpu)
plot_estimate_gpu(grouped_info_list_gpu)
#plot_bandwidth_gpu(grouped_info_list_1gpu)
