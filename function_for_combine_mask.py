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
    # create dictionary for needed seg_data 25 organs
    # found_files = {"liver.nii.gz": None, "colon.nii.gz": None, "heart.nii.gz": None, "small_bowel.nii.gz": None,
    #                "lung_lower_lobe_right.nii.gz": None, "lung_lower_lobe_left.nii.gz": None,
    #                "lung_middle_lobe_right.nii.gz": None, "stomach.nii.gz": None, "pancreas.nii.gz": None,
    #                "duodenum.nii.gz": None, "kidney_right.nii.gz": None, "aorta.nii.gz": None,
    #                "inferior_vena_cava.nii.gz": None, "portal_vein_and_splenic_vein.nii.gz": None,
    #                "gallbladder.nii.gz": None, "urinary_bladder.nii.gz": None,
    #                "kidney_left.nii.gz": None, "spleen.nii.gz": None, "iliac_artery_left.nii.gz": None,
    #                "iliac_artery_right.nii.gz": None, "iliac_vena_right.nii.gz": None, "iliac_vena_left.nii.gz": None,
    #                "sacrum.nii.gz": None, "hip_left.nii.gz": None, "hip_right.nii.gz": None}
    # create dictionary for needed seg_data 25 organs

    found_files = {"liver.nii.gz": None, "colon.nii.gz": None, "heart.nii.gz": None, "small_bowel.nii.gz": None,
                   "stomach.nii.gz": None, "pancreas.nii.gz": None, "kidney_right.nii.gz": None, "aorta.nii.gz": None,
                   "inferior_vena_cava.nii.gz": None, "portal_vein_and_splenic_vein.nii.gz": None,
                   "gallbladder.nii.gz": None, "kidney_left.nii.gz": None, "spleen.nii.gz": None,
                   "iliac_artery_left.nii.gz": None, "iliac_artery_right.nii.gz": None, "iliac_vena_right.nii.gz": None,
                   "iliac_vena_left.nii.gz": None, "urinary_bladder.nii.gz": None}

    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename in found_files:
                file_path = os.path.join(root, filename)
                try:
                    image_data = nib.load(file_path).get_fdata()
                    found_files[filename] = image_data
                except Exception as e:
                    print(f"Error reading {filename}: {e}")

    return found_files


def dilated_mask(combined_mask, affine):

    sitk_out = nib_to_sitk(combined_mask, affine)
    di_filter = sitk.BinaryDilateImageFilter()
    di_filter.SetKernelRadius(5)
    di_filter.SetForegroundValue(1)
    dilated = di_filter.Execute(sitk_out)

    return sitk.GetArrayFromImage(dilated).transpose()


def combine_masks_front(list_front, data_disc):
    # Load the segmentation image
    colon_seg = data_disc["colon.nii.gz"]

    # create empty array for each region
    combined_front_seg = np.zeros(colon_seg.shape, dtype=np.uint8)

    for idx, mask in enumerate(list_front):
        img = data_disc[mask]
        combined_front_seg[img > 0.5] = 1

    return combined_front_seg


def combine_masks_bg(list_bg, combined_final, data_dict, affine, region_number, input_path):
    if region_number == 0:
        list_0 = ["sacrum.nii.gz"]
        for idx, mask in enumerate(list_0):
            img = nib.load(os.path.join(input_path, mask))
            combined_final[img.get_fdata() > 0.5] = 0

    if region_number == 1:
        list_1 = ["lung_lower_lobe_right.nii.gz", "lung_middle_lobe_right.nii.gz",
                  "duodenum.nii.gz"]
        for idx, mask in enumerate(list_1):
            img = nib.load(os.path.join(input_path, mask))
            combined_final[img.get_fdata() > 0.5] = 0

    if region_number == 2:
        list_2 = ["lung_lower_lobe_right.nii.gz", "lung_lower_lobe_left.nii.gz",
                  "lung_middle_lobe_right.nii.gz", "duodenum.nii.gz"]
        for idx, mask in enumerate(list_2):
            img = nib.load(os.path.join(input_path, mask))
            combined_final[img.get_fdata() > 0.5] = 0

    if region_number == 3:
        list_3 = ["lung_lower_lobe_left.nii.gz"]
        for idx, mask in enumerate(list_3):
            img = nib.load(os.path.join(input_path, mask))
            combined_final[img.get_fdata() > 0.5] = 0

    if region_number == 6:
        list_6 = ["hip_left.nii.gz", "hip_right.nii.gz", "sacrum.nii.gz"]
        for idx, mask in enumerate(list_6):
            img = nib.load(os.path.join(input_path, mask))
            combined_final[img.get_fdata() > 0.5] = 0

    # if region_number == 9:
    #     list_9 = ["urinary_bladder.nii.gz"]
    #     for idx, mask in enumerate(list_9):
    #         img = nib.load(os.path.join(input_path, mask))
    #         combined_final[img.get_fdata() > 0.5] = 0

    for idx, mask in enumerate(list_bg):
        img = data_dict[mask]
        combined_final[img > 0.5] = 0

    return nib.Nifti1Image(combined_final, affine)


def process_13_regions_mask(front, background, data_disc, affine, file_out, region_number, input_path, region_6=False):

    if region_6:
        combined_front = combine_masks_front(front, data_disc)
        enlarged = dilated_mask(combined_front, affine)
        # Get the directory part
        directory_part = os.path.dirname(file_out)
        # Desired filename
        desired_filename = "region_0.nii.gz"
        # Create the new path
        seg_0_path = os.path.join(directory_part, desired_filename)
        r0_seg = nib.load(seg_0_path)
        enlarged[r0_seg.get_fdata() > 0.5] = 1
        final_seg = combine_masks_bg(background, enlarged, data_disc, affine, region_number, input_path)
    else:
        combined_front = combine_masks_front(front, data_disc)
        enlarged = dilated_mask(combined_front, affine)
        final_seg = combine_masks_bg(background, enlarged, data_disc, affine, region_number, input_path)

    nib.save(final_seg, file_out)

    return final_seg
