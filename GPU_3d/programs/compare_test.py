import os
import re

def extract_test_number(file_name):
    match = re.match(r'.*test(\d+)_.*', file_name)
    if match:
        return int(match.group(1))
    return None

def extract_numerical_values(filename):
    match = re.match(r'.*width(\d+)_height(\d+)_gpu(\d+)_iter(\d+)_compare(\d+)_overlap(\d+)_.*', filename)
    if match:
        return [int(match.group(i)) for i in range(1, 7)]
    return []

def extract_time(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
        time_match = re.search(r'Time\(event\) - (\d+\.\d+)', content)
        if time_match:
            return float(time_match.group(1))
    return None

def compare_all_times(folder_path):
    # List all files in the directory
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    # Sort the list of files using the custom sorting function
    files = sorted(files, key=extract_numerical_values)

    # Extract partition names from the files
    test_files_by_partition = {}
    for file in files:
        file_path = os.path.join(folder_path, file)
        test_number = extract_test_number(file)
        if test_number is not None:
            partition_key = file.replace(f'_test{test_number}', '')
            if partition_key not in test_files_by_partition:
                test_files_by_partition[partition_key] = []
            test_files_by_partition[partition_key].append(file_path)

    # Compare times for each set of three files
    for partition_key, test_files in test_files_by_partition.items():
        if len(test_files) % 3 == 0:
            for i in range(0, len(test_files), 3):
                test0_file = test_files[i]
                test1_file = test_files[i + 1]
                test2_file = test_files[i + 2]

                width, height = extract_numerical_values(test0_file)[:2]

                time_test0 = extract_time(test0_file)
                time_test1 = extract_time(test1_file)
                time_test2 = extract_time(test2_file)

                if time_test0 is not None and time_test1 is not None and time_test2 is not None:
                    combined_time = time_test1 + time_test2
                    difference = combined_time - time_test0

                    print(f'Width: {width}, Height: {height}, Test0: {time_test0}, Test1 + Test2: {combined_time}, Difference: {difference}')
                else:
                    print(f'Error extracting time from files in partition: {partition_key}')

# Example usage:
folder_path = 'output/'
compare_all_times(folder_path)
