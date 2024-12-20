import argparse
from pathlib import Path
# import nibabel as nib
from get_regions import regions_generation
import os
import dicom2nifti


def get_args_parser():

    parser = argparse.ArgumentParser(description="Combine organs' masks into 13 regions.")

    parser.add_argument("-i", metavar="filepath", dest="input",
                        help="CT nifti image or folder of dicom slices",
                        type=lambda p: Path(p).absolute(), required=True)

    parser.add_argument("-o", metavar="directory", dest="output",
                        help="Output directory for segmentation masks",
                        type=lambda p: Path(p).absolute(), required=True)

    return parser


def process_dicom_and_copy_folders(input_folder, output_folder):
    # Check if the input folder exists
    if not os.path.exists(input_folder):
        print(f"Input folder '{input_folder}' does not exist.")
        return

    # Create the output folder
    os.makedirs(output_folder, exist_ok=True)

    for root, dirs, files in os.walk(input_folder):
        # Create the corresponding directory structure in the output folder
        relative_path = os.path.relpath(root, input_folder)
        output_path = os.path.join(output_folder, relative_path)
        os.makedirs(output_path, exist_ok=True)

        for file in files:
            if file.endswith(".dcm"):
                converted_data_save_dir = os.path.join(output_path, "dicom_to_nifti")
                dcm_to_nifti(root, converted_data_save_dir)
                nii_path = os.path.join(output_path, "dicom_to_nifti.nii")
                save_dir = os.path.join(output_path, "segmentations")
                regions_generation(nii_path, save_dir, output_path)
                print(f"Processed data have been saved to '{output_path}'.")
                break  # Add the folder once and move on to the next


def dcm_to_nifti(input_path, output_path, verbose=False):
    """
    Uses dicom2nifti package (also works on windows)

    input_path: a directory of dicom slices
    output_path: a nifti file path
    """

    dicom2nifti.dicom_series_to_nifti(input_path, output_path, reorient_nifti=True)


def main(segmentations=None):
    parser = get_args_parser()
    args = parser.parse_args()
    input_folder = Path(args.input)
    output_folder = Path(args.output)

    process_dicom_and_copy_folders(input_folder, output_folder)


if __name__ == "__main__":
    main()
