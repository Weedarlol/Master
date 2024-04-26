import os
from twodimentional import twodimentional_plot
from threedimentional import threedimentional_plot

# Define available plots
available_plots = {
    '2d': ['CPU Computation time vs Theoretical Time', 'Overlap vs No Overlap', 'Computational time: Communication vs Computation', 'GPU Computation time vs Theoretical Time'],  # List your 2D plot options here
    '3d': ['CPU Computation time vs Theoretical Time', 'CPU Computation, Nodes', 'Overlap vs No Overlap', 'Computational time: Communication vs Computation', 'GPU Computation time vs Theoretical Time']   # List your 3D plot options here
}

def load_previous_settings():
    settings = []
    with open("Previous_Settings.txt", "r") as f:
        for line in f:
            settings.append(line.strip())  # strip() removes the newline character at the end of each line
    return settings

def save_settings(settings):
    with open("Previous_Settings.txt", "w") as f:
        for setting in settings:
            f.write(str(setting) + "\n")

def select_plots(previous_settings):
    settings = []
    while True:
        plot_type = input("Select plot type (2d/3d/prev): ").lower()
        if plot_type in available_plots:
            settings.append(plot_type)
            plots = available_plots[plot_type]
            print("Available plots:")
            for i, plot in enumerate(plots, 1):
                print(f"{i}. {plot}")
            plot_indices = input("Select plot(s) (comma-separated numbers or 'all'): ")
            if plot_indices.lower() == 'all':
                settings.extend(plots)
                return settings
            else:
                indices = plot_indices.split(',')
                selected_plots = [plots[int(idx.strip()) - 1] for idx in indices]
                settings.extend(selected_plots)
                return settings
            
        elif plot_type == "prev":
            return previous_settings
            
        else:
            print("Invalid plot type. Please select either '2d', '3d' or \"prev\"ious settings.")

def save_plots(plots):
    save_option = input("Do you want to save the plots? (yes/no): ").lower()
    with open("Previous_Settings.txt", "r") as f:
        dimention = f.readline().strip()

    info_list_gpu_allocated = False
    if dimention == "2d":
        folder_path_ex3_cpu = "../ex3/CPU_2d/output"
        folder_path_fox_cpu = "../fox/CPU_2d/output"
        folder_path_ex3_gpu = "../ex3/GPU_2d/output"
        folder_path_fox_gpu = "../fox/GPU_2d/output"
        folder_path_ex3_1gpu = "../ex3/GPU_2d_1GPU/output"
        folder_path_fox_1gpu = "../fox/GPU_2d_1GPU/output"

        for plot in plots[1:]:
            if plot == "CPU Computation time vs Theoretical Time":
                info_list_cpu = []
                info_list_cpu = twodimentional_plot.process_files(folder_path_ex3_cpu, info_list_cpu)
                info_list_cpu = twodimentional_plot.process_files(folder_path_fox_cpu, info_list_cpu)
                grouped_info_list_cpu = twodimentional_plot.group_by_string(info_list_cpu)
                twodimentional_plot.plot_info_cpu(grouped_info_list_cpu, save_option)
            elif plot == "Overlap vs No Overlap":
                if not info_list_gpu_allocated:
                    info_list_gpu = []
                    info_list_gpu = twodimentional_plot.process_files(folder_path_ex3_gpu, info_list_gpu)
                    info_list_gpu_allocated = True
                    grouped_info_list_gpu = twodimentional_plot.group_by_string(info_list_gpu)
                twodimentional_plot.plot_overlap_gpu(grouped_info_list_gpu, save_option)
            elif plot == "Computational time: Communication vs Computation":
                if not info_list_gpu_allocated:
                    info_list_gpu = []
                    info_list_gpu = twodimentional_plot.process_files(folder_path_ex3_gpu, info_list_gpu)
                    info_list_gpu_allocated = True
                    grouped_info_list_gpu = twodimentional_plot.group_by_string(info_list_gpu)
                twodimentional_plot.plot_estimate_gpu(grouped_info_list_gpu, save_option)
            elif plot == "GPU Computation time vs Theoretical Time":
                info_list_1gpu = []
                info_list_1gpu = twodimentional_plot.process_files(folder_path_ex3_1gpu, info_list_1gpu)
                grouped_info_list_1gpu = twodimentional_plot.group_by_string(info_list_1gpu)
                twodimentional_plot.plot_bandwidth_gpu(grouped_info_list_1gpu, save_option)
    elif dimention == "3d":
        folder_path_ex3_cpu = "../ex3/CPU_3d/output"
        folder_path_fox_cpu = "../fox/CPU_3d/output"
        folder_path_ex3_gpu = "../ex3/GPU_3d/output"
        folder_path_ex3_1gpu = "../ex3/GPU_3d_1GPU/output"
        folder_path_fox_gpu = "../fox/GPU_3d/output"
        folder_path_ex3_cpu_cpu = "../ex3/CPU_CPU/output"
        folder_path_fox_1gpu = "../fox/GPU_3d_1GPU/output"
        
        for plot in plots[1:]:
            if plot == "CPU Computation time vs Theoretical Time":
                info_list_cpu = []
                info_list_cpu = threedimentional_plot.process_files(folder_path_ex3_cpu, info_list_cpu)
                info_list_cpu = threedimentional_plot.process_files(folder_path_fox_cpu, info_list_cpu)
                grouped_info_list_cpu = threedimentional_plot.group_by_string(info_list_cpu)
                threedimentional_plot.plot_info_cpu(grouped_info_list_cpu, save_option)
            elif plot == "CPU Computation, Nodes":
                if not info_list_gpu_allocated:
                    info_list_cpu = []
                    info_list_cpu = threedimentional_plot.process_files(folder_path_ex3_cpu_cpu, info_list_cpu)
                    grouped_info_list_cpu = threedimentional_plot.group_by_string(info_list_cpu)
                threedimentional_plot.plot_info_cpu_cpu(grouped_info_list_cpu, save_option)
            elif plot == "Overlap vs No Overlap":
                if not info_list_gpu_allocated:
                    info_list_gpu = []
                    info_list_gpu = threedimentional_plot.process_files(folder_path_ex3_gpu, info_list_gpu)
                    info_list_gpu_allocated = True
                    grouped_info_list_gpu = threedimentional_plot.group_by_string(info_list_gpu)
                threedimentional_plot.plot_overlap_gpu(grouped_info_list_gpu, save_option)
            elif plot == "Computational time: Communication vs Computation":
                if not info_list_gpu_allocated:
                    info_list_gpu = []
                    info_list_gpu = threedimentional_plot.process_files(folder_path_ex3_gpu, info_list_gpu)
                    info_list_gpu_allocated = True
                    grouped_info_list_gpu = threedimentional_plot.group_by_string(info_list_gpu)
                threedimentional_plot.plot_estimate_gpu(grouped_info_list_gpu, save_option)
            elif plot == "GPU Computation time vs Theoretical Time":
                info_list_1gpu = []
                info_list_1gpu = threedimentional_plot.process_files(folder_path_ex3_1gpu, info_list_1gpu)
                grouped_info_list_1gpu = threedimentional_plot.group_by_string(info_list_1gpu)
                threedimentional_plot.plot_bandwidth_gpu(grouped_info_list_1gpu, save_option)

def main():
    print("This requires you to have run the required programs first.")
    previous_settings = load_previous_settings()
    print("Previous settings loaded.")
    plot_list = select_plots(previous_settings)
    save_plots(plot_list)
    save_settings(plot_list)
    print("Settings stored.")

if __name__ == "__main__":
    main()
