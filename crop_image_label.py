from code_test import crop_image
import nibabel as nib


def mian():
    label_path = "./masks/13_regions/knn_part.nii.gz"
    image_path = "./masks/201 THX-ABD AX 3-3 iDose (3)_3.nii.gz"
    label = nib.load(label_path)
    image = nib.load(image_path)

    label_data = label.get_fdata()
    image_data = image.get_fdata()
    # crop the image
    cropped_label_data, start_point, end_point = crop_image(label_data)
    cropped_image = image_data[end_point[0]:start_point[0],
                               end_point[1]:start_point[1],
                               end_point[2]:start_point[2]]
    # Visualize the cropped image and label
    # load_and_visualize_nifti_2d(cropped_image, cropped_label_data)
    # # Save the new NIfTI file
    label_nifti = nib.Nifti1Image(cropped_label_data, label.affine)
    image_nifti = nib.Nifti1Image(cropped_image, image.affine)
    nib.save(label_nifti, "./data/croped_scan/cropped_label.nii.gz")
    nib.save(image_nifti, "./data/croped_scan/cropped_image.nii.gz")


if __name__ == "__main__":
    mian()
