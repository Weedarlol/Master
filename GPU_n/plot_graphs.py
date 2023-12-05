import os
import matplotlib.pyplot as plt

def parse_file(file_name):
    parts = file_name.split("_")
    width = int(parts[0].replace("width", ""))
    height = int(parts[1].replace("height", ""))
    gpu = int(parts[2].replace("gpu", ""))
    iterations = int(parts[3].replace("iter", ""))
    overlap = bool(int(parts[4].replace("overlap", "")))
    partition = parts[5].replace(".out", "")

    with open(os.path.join("output", file_name), 'r') as file:
        content = file.readline().strip()
        time = float(content.split(", ")[0].split(" - ")[1])
        solution = content.split(", ")[1].split(" - ")[1] == "Yes"
        iterations_computed = int(content.split(", ")[2].split(" - ")[1])
    
    return width, height, gpu, iterations, overlap, partition, time, solution, iterations_computed


file_list = os.listdir("output")

results = {}

for file_name in file_list:
    result = parse_file(file_name)
    # Grouping results based on width, height, and partition
    key = (result[0], result[1], result[5])
    if key in results:
        results[key].append(result)
    else:
        results[key] = [result]

# Generating line plots
for key, result_list in results.items():
    plt.figure()
    plt.title(f"Width: {key[0]}, Height: {key[1]}, Partition: {key[2]}")
    for result in result_list:
        iterations = result[3]
        time = result[6]
        overlap = result[4]
        if overlap:
            label = "Overlap: True"
        else:
            label = "Overlap: False"
        plt.plot(iterations, time, label=label)
    plt.xlabel("Iterations")
    plt.ylabel("Time")
    plt.legend()
    plt.show()
