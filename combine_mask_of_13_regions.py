# from pathlib import Path
import os
import numpy as np
import nibabel as nib
import SimpleITK as sitk


def nib_to_sitk(data, affine):
    flip_xy = np.diag((-1, -1, 1))
    origin = np.dot(flip_xy, affine[:3, 3]).astype(np.float64)
    rzs = affine[:3, :3]
    spacing = np.sqrt(np.sum(rzs * rzs, axis=0))
    r = rzs / spacing
    direction = np.dot(flip_xy, r).flatten()
    image = sitk.GetImageFromArray(data.transpose())
    image.SetOrigin(origin)
    image.SetSpacing(spacing)
    image.SetDirection(direction)
    return image


# Function to search for files in an unknown directory
def find_and_read_nifti_data(directory):
    # create dictionary for needed seg_data
    found_files = {"liver.nii.gz": None, "colon.nii.gz": None, "heart.nii.gz": None, "small_bowel.nii.gz": None,
                   "lung_lower_lobe_right.nii.gz": None, "lung_lower_lobe_left.nii.gz": None,
                   "lung_middle_lobe_right.nii.gz": None, "stomach.nii.gz": None, "pancreas.nii.gz": None,
                   "duodenum.nii.gz": None, "kidney_right.nii.gz": None, "aorta.nii.gz": None,
                   "inferior_vena_cava.nii.gz": None, "portal_vein_and_splenic_vein.nii.gz": None,
                   "gallbladder.nii.gz": None, "urinary_bladder.nii.gz": None, "heart_ventricle_right.nii.gz": None,
                   "kidney_left.nii.gz": None, "spleen.nii.gz": None, "iliac_artery_left.nii.gz": None,
                   "iliac_artery_right.nii.gz": None, "iliac_vena_right.nii.gz": None, "iliac_vena_left.nii.gz": None,
                   "sacrum.nii.gz": None, "hip_left.nii.gz": None, "hip_right.nii.gz": None}

    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename in found_files:
                file_path = os.path.join(root, filename)
                try:
                    img = nib.load(file_path)
                    data = img.get_fdata()
                    found_files[filename] = data
                except Exception as e:
                    print(f"Error reading {filename}: {e}")

    return found_files, img.affine


def dilated_mask(combined_mask, affine):

    sitk_out = nib_to_sitk(combined_mask, affine)
    di_filter = sitk.BinaryDilateImageFilter()
    di_filter.SetKernelRadius(5)
    di_filter.SetForegroundValue(1)
    dilated = di_filter.Execute(sitk_out)

    return sitk.GetArrayFromImage(dilated).transpose()


def combine_masks_front(list_front, combined_front_seg, data_disc):

    for idx, mask in enumerate(list_front):
        img = data_disc[mask]
        combined_front_seg[img > 0.5] = 1

    return combined_front_seg


def combine_masks_bg(list_bg, combined_final, data_disc, affine):

    for idx, mask in enumerate(list_bg):
        img = data_disc[mask]
        combined_final[img > 0.5] = 0

    return nib.Nifti1Image(combined_final, affine)


def process_13_regions_mask(front, background, data_disc, combined, affine, file_out):

    combined_front = combine_masks_front(front, combined, data_disc)
    enlarged = dilated_mask(combined_front, affine)
    final_seg = combine_masks_bg(background, enlarged, data_disc, affine)

    nib.save(final_seg, file_out)

    return final_seg


def main():
    # Replace 'directory_path' with the path to the directory where you want to search for the files.
    directory_path = "segmentation"  # change to parse output folder
    found_data, affine = find_and_read_nifti_data(directory_path)

    # Load the colon segmentation image
    colon_seg = found_data["colon.nii.gz"]
    liver_seg = found_data["liver.nii.gz"]
    spleen_seg = found_data["spleen.nii.gz"]
    small_bowel_seg = found_data["small_bowel.nii.gz"]

    combined_r0_r6 = np.zeros(colon_seg.shape, dtype=np.uint8)
    combined_r1_r2 = np.zeros(liver_seg.shape, dtype=np.uint8)
    combined_r3 = np.zeros(spleen_seg.shape, dtype=np.uint8)
    combined_r9 = np.zeros(small_bowel_seg.shape, dtype=np.uint8)
    # List the front and background masks for region 1
    r0_front = ["colon.nii.gz"]
    r1_front = ["gallbladder.nii.gz", "liver.nii.gz"]
    r2_front = ["stomach.nii.gz", "pancreas.nii.gz", "liver.nii.gz"]
    r3_front = ["spleen.nii.gz"]
    r6_front = ["urinary_bladder.nii.gz", "colon.nii.gz"]
    r9_front = ["small_bowel.nii.gz"]

    r0_bg = [
        "liver.nii.gz",
        "stomach.nii.gz",
        "urinary_bladder.nii.gz",
        "kidney_right.nii.gz",
        "kidney_left.nii.gz",
        "small_bowel.nii.gz",
        "spleen.nii.gz",
        "gallbladder.nii.gz",
        "pancreas.nii.gz",
        "aorta.nii.gz",
        "inferior_vena_cava.nii.gz",
        "portal_vein_and_splenic_vein.nii.gz",
        "iliac_artery_left.nii.gz",
        "iliac_artery_right.nii.gz",
        "iliac_vena_left.nii.gz",
        "iliac_vena_right.nii.gz",
        "sacrum.nii.gz"
    ]
    r1_bg = [
        "colon.nii.gz",
        "small_bowel.nii.gz",
        "lung_lower_lobe_right.nii.gz",
        "lung_middle_lobe_right.nii.gz",
        "stomach.nii.gz",
        "pancreas.nii.gz",
        "duodenum.nii.gz",
        "kidney_right.nii.gz",
        "aorta.nii.gz",
        "inferior_vena_cava.nii.gz",
        "portal_vein_and_splenic_vein.nii.gz",
        # "heart_ventricle_right.nii.gz",
        "heart.nii.gz"
    ]
    r2_bg = [
        "colon.nii.gz",
        "small_bowel.nii.gz",
        "lung_lower_lobe_right.nii.gz",
        "lung_lower_lobe_left.nii.gz",
        "lung_middle_lobe_right.nii.gz",
        "duodenum.nii.gz",
        "spleen.nii.gz",
        "kidney_left.nii.gz",
        "aorta.nii.gz",
        "inferior_vena_cava.nii.gz",
        "portal_vein_and_splenic_vein.nii.gz",
        "heart.nii.gz"
    ]
    r3_bg = [
        "stomach.nii.gz",
        "kidney_left.nii.gz",
        "aorta.nii.gz",
        "inferior_vena_cava.nii.gz",
        "portal_vein_and_splenic_vein.nii.gz",
        "heart.nii.gz"
    ]

    r6_bg = [
        "small_bowel.nii.gz",
        "hip_left.nii.gz",
        "hip_right.nii.gz",
        "aorta.nii.gz",
        "inferior_vena_cava.nii.gz",
        "liver.nii.gz",
        "stomach.nii.gz",
        "kidney_right.nii.gz",
        "kidney_left.nii.gz",
        "small_bowel.nii.gz",
        "spleen.nii.gz",
        "gallbladder.nii.gz",
        "pancreas.nii.gz",
        "portal_vein_and_splenic_vein.nii.gz",
        "iliac_artery_left.nii.gz",
        "iliac_artery_right.nii.gz",
        "iliac_vena_left.nii.gz",
        "iliac_vena_right.nii.gz",
        "sacrum.nii.gz"
    ]
    r9_bg = [
        "iliac_artery_left.nii.gz",
        "iliac_artery_right.nii.gz",
        "iliac_vena_left.nii.gz",
        "iliac_vena_right.nii.gz",
        "colon.nii.gz",
        "stomach.nii.gz",
        "gallbladder.nii.gz",
        "urinary_bladder.nii.gz",
        "spleen.nii.gz",
        "liver.nii.gz",
        "kidney_right.nii.gz",
        "kidney_left.nii.gz",
        "pancreas.nii.gz",
        "aorta.nii.gz",
        "inferior_vena_cava.nii.gz",
        "portal_vein_and_splenic_vein.nii.gz"
    ]

    r0_seg = process_13_regions_mask(r0_front, r0_bg, found_data, combined_r0_r6,
                                     affine, "13_regions_output/region_0.nii.gz")
    _ = process_13_regions_mask(r1_front, r1_bg, found_data, combined_r1_r2,
                                affine, "13_regions_output/region_1.nii.gz")
    _ = process_13_regions_mask(r2_front, r2_bg, found_data, combined_r1_r2,
                                affine, "13_regions_output/region_2.nii.gz")
    _ = process_13_regions_mask(r3_front, r3_bg, found_data, combined_r3,
                                affine, "13_regions_output/region_3.nii.gz")
    _ = process_13_regions_mask(r6_front, r6_bg, found_data, combined_r0_r6,
                                affine, "13_regions_output/region_6.nii.gz")
    r9_seg = process_13_regions_mask(r9_front, r9_bg, found_data, combined_r9,
                                     affine, "13_regions_output/region_9.nii.gz")

    # Define the regions that same as r0 and r9
    r0_region = [4, 5, 7, 8]
    r9_region = [10, 11, 12]

    for region in r0_region:
        filename = f"13_regions_output/region_{region}.nii.gz"
        nib.save(r0_seg, filename)

    for region in r9_region:
        filename = f"13_regions_output/region_{region}.nii.gz"
        nib.save(r9_seg, filename)


if __name__ == "__main__":
    main()