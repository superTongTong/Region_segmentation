from region_segmentation import dcm_to_nifti
import os


def start_dicom_to_nifti(input_folder, output_folder):
    count = 0
    for root, dirs, files in os.walk(input_folder):
        # Create the corresponding directory structure in the output folder
        # relative_path = os.path.relpath(root, input_folder)
        # output_path = os.path.join(output_folder, relative_path)
        # os.makedirs(output_path, exist_ok=True)

        for file in files:
            if file.endswith(".dcm"):
                name = f"{file[:5]}"
                converted_data_save_dir = os.path.join(output_folder, name)
                dcm_to_nifti(root, converted_data_save_dir)
                count += 1
                print(f"file {name} have been processed.{count}")
                break  # Add the folder once and move on to the next


if __name__ == "__main__":
    input_folder = "E:/graduation_project_TUe/data_from_Lotte/3_region_reg/PM_scans_first60_1mm/22222"
    output_folder = "E:/graduation_project_TUe/data_from_Lotte/3_region_reg/PM_scans_first60_1mm_nifti"
    start_dicom_to_nifti(input_folder, output_folder)
