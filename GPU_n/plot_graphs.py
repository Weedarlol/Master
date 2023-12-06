import os
import matplotlib.pyplot as plt

output_folder = "output"  # Specify the output folder path

# Function to extract variables from document titles
# Function to extract variables from document titles
def extract_variables_from_title(title):
    title_parts = title.split("_")
    variables = {}
    variables["Width"] = int(title_parts[0].replace("width", ""))
    variables["Height"] = int(title_parts[1].replace("height", ""))
    variables["Gpu"] = int(title_parts[2].replace("gpu", ""))
    variables["Overlap"] = bool(int(title_parts[-2].replace("overlap", "")))
    variables["Partition"] = title_parts[-1].replace(".out", "")
    return variables


# Function to extract variables from document contents
def extract_variables_from_content(content):
    lines = content.strip().split("\n")
    variable_names = ["Time", "Iter", "IterComputed", "Finished"]
    variables = {}
    for name in variable_names:
        variables[name] = []
    for line in lines:
        values = line.split(", ")
        for idx, name in enumerate(variable_names):
            variables[name].append(float(values[idx]))
            if(name == "Finished"):
                if(int(values[idx]) == 1):
                    return variables
            
    return variables

# Function to process a document and extract variables
def process_document(file_path):
    with open(file_path, "r") as file:
        content = file.read()
        title = os.path.splitext(os.path.basename(file_path))[0]
        title_variables = extract_variables_from_title(title)
        content_variables = extract_variables_from_content(content)
        return {**title_variables, **content_variables}

# Main function to process all documents in the output folder
def process_all_documents():
    output_files = os.listdir(output_folder)
    data = []
    for file_name in output_files:
        file_path = os.path.join(output_folder, file_name)
        if file_path.endswith(".out"):
            document_data = process_document(file_path)
            if document_data:
                data.append(document_data)
    return data

# Example usage
result = process_all_documents()



for document_data in result:
    print(document_data)


# Access the required variables for plotting
time = document_data["Time"]
iter_computed = document_data["IterComputed"]
iter_data = document_data["Iter"]

# Create the plot
plt.plot(iter_computed, time, marker="o")
plt.xscale("log")  # Set the X-axis scale to logarithmic
plt.xlabel("IterComputed")
plt.ylabel("Time")
plt.title("Time vs. IterComputed")
plt.grid(True)

# Show the plot
plt.show()