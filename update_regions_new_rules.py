import argparse
from pathlib import Path
import nibabel as nib
import os
import SimpleITK as sitk
import numpy as np
from function_for_combine_mask import dilated_mask
import shutil
from tqdm import tqdm
import time


def get_args_parser():

    parser = argparse.ArgumentParser(description="Combine organs' masks into 13 regions.")

    parser.add_argument("-i", metavar="filepath", dest="input",
                        help="CT nifti image or folder of dicom slices",
                        type=lambda p: Path(p).absolute(), required=True)

    return parser


def process_r2r3_regions(front_list, bg_list, data_path, r_number):
    # Load the segmentation image
    colon_seg = nib.load(os.path.join(data_path, "colon.nii.gz"))

    # create empty array for each region
    combined_front_seg = np.zeros(colon_seg.shape, dtype=np.uint8)

    # Process front_list
    for mask_ft in tqdm(front_list, desc=f"Processing front_layer of {r_number}"):
        img = nib.load(os.path.join(data_path, mask_ft))
        combined_front_seg[img.get_fdata() > 0.5] = 1

    enlarged = dilated_mask(combined_front_seg, colon_seg.affine)

    # Process bg_list
    for mask_bg in tqdm(bg_list, desc=f"Processing background_layer of {r_number}"):
        img = nib.load(os.path.join(data_path, mask_bg))
        enlarged[img.get_fdata() > 0.5] = 0

    return nib.Nifti1Image(enlarged, colon_seg.affine)


def update_r2r3_regions_copy_other_regions(data_path, out_path):
    print("start update region 2 and 3....")
    r2_front = ["liver.nii.gz"]
    r2_bg = [
        "colon.nii.gz",
        "small_bowel.nii.gz",
        "spleen.nii.gz",
        "kidney_left.nii.gz",
        "aorta.nii.gz",
        "inferior_vena_cava.nii.gz",
        "portal_vein_and_splenic_vein.nii.gz",
        "lung_lower_lobe_right.nii.gz",
        "lung_lower_lobe_left.nii.gz",
        "lung_middle_lobe_right.nii.gz",
        "duodenum.nii.gz",
        "heart.nii.gz"
    ]
    new_r2 = process_r2r3_regions(r2_front, r2_bg, data_path, "region_2")
    nib.save(new_r2, os.path.join(out_path, "updated_region_2.nii.gz"))

    r3_front = ["stomach.nii.gz", "pancreas.nii.gz", "spleen.nii.gz"]
    r3_bg = [
        "kidney_left.nii.gz",
        "aorta.nii.gz",
        "inferior_vena_cava.nii.gz",
        "portal_vein_and_splenic_vein.nii.gz",
        "colon.nii.gz",
        "small_bowel.nii.gz",
        "lung_lower_lobe_left.nii.gz",
        "heart.nii.gz"
    ]
    new_r3 = process_r2r3_regions(r3_front, r3_bg, data_path, "region_3")
    nib.save(new_r3, os.path.join(out_path, "updated_region_3.nii.gz"))
    print(f"saving updated region 2 and region 3 data.")


def combine_extra_structure(in_path, combine_list, np_arrays, text, itk_img=None):

    for root, dire, files in os.walk(in_path):

        for file in tqdm(combine_list, desc=f"combining {text}"):
            if file in files:
                converted_data_save_dir = os.path.join(root, file)
                itk_img = sitk.ReadImage(converted_data_save_dir, sitk.sitkInt8)
                np_img = sitk.GetArrayFromImage(itk_img)
                np_arrays.append(np_img)

    return np_arrays, itk_img


def combine_new_13_regions(seg_path, save_dir):
    print("start combine 13 regions mask....")
    np_arrays = []
    seg_list = [
        "hip_left.nii.gz",
        "hip_right.nii.gz",
        "kidney_right.nii.gz",
        "kidney_left.nii.gz",
        "iliac_artery_left.nii.gz",
        "iliac_artery_right.nii.gz",
        "iliac_vena_left.nii.gz",
        "iliac_vena_right.nii.gz",
        "duodenum.nii.gz",
        "lung_lower_lobe_left.nii.gz",
        "lung_lower_lobe_right.nii.gz",
        "lung_middle_lobe_right.nii.gz",
        "aorta.nii.gz",
        "inferior_vena_cava.nii.gz",
        "portal_vein_and_splenic_vein.nii.gz",
        "heart.nii.gz"
    ]

    regions_list = [
        "region_0.nii.gz",
        "region_1.nii.gz",
        "updated_region_2.nii.gz",
        "updated_region_3.nii.gz",
        "region_4.nii.gz",
        "region_5.nii.gz",
        "region_6.nii.gz",
        "region_7.nii.gz",
        "region_8.nii.gz",
        "region_9.nii.gz",
        "region_10.nii.gz",
        "region_11.nii.gz",
        "region_12.nii.gz"
    ]

    region_arrays, _ = combine_extra_structure(save_dir, regions_list, np_arrays, "13 regions")
    final_seg_arrays, itk_img = combine_extra_structure(seg_path, seg_list, region_arrays, "extra structures")

    # Stack the numpy arrays along the last dimension (axis=-1)
    combined = np.stack(final_seg_arrays, axis=-1)

    combined_itk = sitk.GetImageFromArray(combined, isVector=True)

    combined_itk.CopyInformation(itk_img)
    sitk.WriteImage(combined_itk, os.path.join(save_dir, 'combined_13_regions_with_more_structures.nii.gz'))
    print("13 regions mask combination is complete.")


def find_dir_folders(input_folder):
    # Check if the input folder exists
    if not os.path.exists(input_folder):
        print(f"Input folder '{input_folder}' does not exist.")
        return
    start_time = time.time()  # Record the start time
    for root, dirs, files in os.walk(input_folder):
        if "13_regions" in dirs:
            new_save_dir = os.path.join(root, "updated_13_regions")
            os.makedirs(new_save_dir, exist_ok=True)
            files_to_copy = [
                            "region_0.nii.gz",
                            "region_1.nii.gz",
                            "region_4.nii.gz",
                            "region_5.nii.gz",
                            "region_6.nii.gz",
                            "region_7.nii.gz",
                            "region_8.nii.gz",
                            "region_9.nii.gz",
                            "region_10.nii.gz",
                            "region_11.nii.gz",
                            "region_12.nii.gz"]
            for file_name in files_to_copy:
                source = os.path.join(root, "13_regions")
                source_path = os.path.join(source, file_name)
                destination_path = os.path.join(new_save_dir, file_name)
                shutil.copy2(source_path, destination_path)
        if "segmentations" in dirs:
            seg_path = os.path.join(root, "segmentations")
            update_r2r3_regions_copy_other_regions(seg_path, new_save_dir)
            combine_new_13_regions(seg_path, new_save_dir)
            print(f"Processed data have been saved to '{new_save_dir}'.")
            # break  # Only process the first folder
    end_time = time.time()  # Record the end time
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds.")


def main():
    parser = get_args_parser()
    args = parser.parse_args()
    input_folder = Path(args.input)

    find_dir_folders(input_folder)


if __name__ == "__main__":
    main()
