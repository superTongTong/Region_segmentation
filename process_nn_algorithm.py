from nn_algorithm import closing_image, find_non_overlap, divide_overlapped_region
import time
import SimpleITK as sitk
import os
from tqdm import tqdm
import argparse


def nearest_neighbour_process(folder_in, folder_out):
    # Check if the input folder exists
    if not os.path.exists(folder_in):
        raise FileNotFoundError(f"The input folder '{folder_in}' does not exist.")
    # Create the output folder if it doesn't exist
    os.makedirs(folder_out, exist_ok=True)

    ''''here call the function from nn_algorithm.py to process the data'''
    # List all files in the input folder
    nii_files = os.listdir(folder_in)

    for file in tqdm(nii_files):
        # if file.endswith('.nii'):
        suffixes = ('.nii', '.nii.gz')
        if file.endswith(suffixes):
            sitk_orig = sitk.ReadImage(f"{folder_in}/{file}", sitk.sitkInt8)
            array_orig = sitk.GetArrayFromImage(sitk_orig)
            c_image, t_image = closing_image(sitk_orig, kernel_radius=[3, 3, 3])
            non_overlapped_voxel = find_non_overlap(c_image, t_image)

            processed_image = divide_overlapped_region(non_overlapped_voxel, array_orig, 10)

            # covert the processed image to sitk image
            img_for_save = sitk.GetImageFromArray(processed_image)
            img_for_save.CopyInformation(sitk_orig)
            save_dir = f"{folder_out}/after_nn"
            os.makedirs(save_dir, exist_ok=True)

            # Export Image
            sitk.WriteImage(img_for_save, f"{save_dir}/{file}")

def main(args):
    # input_path = 'data/nnunet/raw/Dataset018_Orig_nn/labelsTr'
    # output_path = 'data/nnunet/raw/Dataset018_Orig_nn/'
    nearest_neighbour_process(args.input, args.output)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input', default=r"data/nnunet/raw/Dataset018_Orig_nn/labelsTr", type=str,
                        help='dicom input folder path (default: data/nnunet/raw/Dataset018_Orig_nn/labelsTr)')
    parser.add_argument('-o', '--output',
                        default=r"data/nnunet/raw/Dataset018_Orig_nn/",
                        type=str, help='NIfTI output folder path (default: data/nnunet/raw/Dataset018_Orig_nn/)')
    start_time = time.time()
    args = parser.parse_args()
    main(args)
    total_time = time.time() - start_time
    print(f"--- {total_time:.2f} seconds ---")