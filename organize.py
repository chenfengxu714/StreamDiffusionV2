import shutil
import os


input_folder = "outputs"
output_folder = "outputs_organized"

if not os.path.exists(output_folder):
    os.makedirs(output_folder, exist_ok=True)

for case_name in os.listdir(input_folder):
    os.makedirs(os.path.join(output_folder, case_name), exist_ok=True)
    for name in os.listdir(os.path.join(input_folder, case_name)):
        source_file = os.path.join(input_folder, case_name, name, "output_000.mp4")
        destination_file = os.path.join(output_folder, case_name, name + ".mp4")
        shutil.copy(source_file, destination_file)
