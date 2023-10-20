import SimpleITK as sitk
import numpy as np

# itk_img_0 = sitk.ReadImage('./masks/13_regions/region_0.nii.gz', sitk.sitkInt8)
# itk_img_1 = sitk.ReadImage('./masks/13_regions/region_1.nii.gz', sitk.sitkInt8)
# itk_img_2 = sitk.ReadImage('./masks/13_regions/region_2.nii.gz', sitk.sitkInt8)
#
# np_img_0 = sitk.GetArrayFromImage(itk_img_0)
# np_img_1 = sitk.GetArrayFromImage(itk_img_1)
# np_img_2 = sitk.GetArrayFromImage(itk_img_2)
#
# combined = np.stack([
# np_img_0,
# np_img_1,
# np_img_2,
# ], axis=-1)
#
# combined_itk = sitk.GetImageFromArray(combined, isVector=True)
#
# combined_itk.CopyInformation(itk_img_0)
#
# sitk.WriteImage(combined_itk, 'masks/combined_13_regions.nii.gz')

np_arrays = []
itk_img = None
for i in range(3):
    itk_img = sitk.ReadImage(f'./masks/13_regions/region_{i}.nii.gz', sitk.sitkInt8)
    np_img = sitk.GetArrayFromImage(itk_img)
    np_arrays.append(np_img)

# Stack the numpy arrays along the last dimension (axis=-1)
combined = np.stack(np_arrays, axis=-1)

combined_itk = sitk.GetImageFromArray(combined, isVector=True)

combined_itk.CopyInformation(itk_img)
sitk.WriteImage(combined_itk, 'test.nii.gz')
