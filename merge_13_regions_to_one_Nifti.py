import SimpleITK as sitk
import numpy as np
import os
'''
    This code is combining 6 regions into one Nifti file for KNN test.
    The combined regions are: 0 1 2 3 6 9.
    HAVE TO ADD BACKGROUND ON TOP OF THIS
'''
data_path = './processed_data_v4/00010/1/segmentations'
np_arrays = []
itk_img = None
for root, dirs, files in os.walk(data_path):
    for file in files:
        converted_data_save_dir = os.path.join(root, file)
        itk_img = sitk.ReadImage(converted_data_save_dir, sitk.sitkInt8)
        np_img = sitk.GetArrayFromImage(itk_img)
        np_arrays.append(np_img)
# combined_list = [0, 1, 2, 3, 6, 9]
# for i in combined_list:
#     itk_img = sitk.ReadImage(f'./masks/13_regions/region_{i}.nii.gz', sitk.sitkInt8)
#     np_img = sitk.GetArrayFromImage(itk_img)
#     np_arrays.append(np_img)

# Stack the numpy arrays along the last dimension (axis=-1)
combined = np.stack(np_arrays, axis=-1)

combined_itk = sitk.GetImageFromArray(combined, isVector=True)

combined_itk.CopyInformation(itk_img)
sitk.WriteImage(combined_itk, './processed_data_v4/00010/1/combined_all_segs.nii.gz')
