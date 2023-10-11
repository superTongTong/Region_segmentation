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

                    found_files[filename] = img.get_fdata()
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


def combine_masks_front(list_front, data_disc):
    # Load the segmentation image
    colon_seg = data_disc["colon.nii.gz"]

    # create empty array for each region
    combined_front_seg = np.zeros(colon_seg.shape, dtype=np.uint8)

    for idx, mask in enumerate(list_front):
        img = data_disc[mask]
        combined_front_seg[img > 0.5] = 1

    return combined_front_seg


def combine_masks_bg(list_bg, combined_final, data_disc, affine):

    for idx, mask in enumerate(list_bg):
        img = data_disc[mask]
        combined_final[img > 0.5] = 0

    return nib.Nifti1Image(combined_final, affine)


def process_13_regions_mask(front, background, data_disc, affine, file_out):

    combined_front = combine_masks_front(front, data_disc)
    enlarged = dilated_mask(combined_front, affine)
    final_seg = combine_masks_bg(background, enlarged, data_disc, affine)

    nib.save(final_seg, file_out)

    return final_seg
