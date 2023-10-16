import nibabel as nib
import numpy as np

'''
Create new python code for merge 13 regions into 1 Nifti file. 
But get issue, that one Nifti can only save 1 layer 3d images, 
which means it can not handle the overlapped regions.
'''

#读取数据路径和储存路径都需要修改。
# Load the needed NIfTI files
nii_0 = nib.load('./masks/13_regions/region_0.nii.gz')
nii_1 = nib.load('./masks/13_regions/region_1.nii.gz')
nii_2 = nib.load('./masks/13_regions/region_2.nii.gz')
nii_3 = nib.load('./masks/13_regions/region_3.nii.gz')
nii_6 = nib.load('./masks/13_regions/region_6.nii.gz')
nii_9 = nib.load('./masks/13_regions/region_9.nii.gz')

# created empty array
combined_data = np.zeros(nii_6.shape, dtype=np.uint8)

# assign regions to difference classes.
combined_data[nii_0.get_fdata()>0.5] = 1
combined_data[nii_1.get_fdata()>0.5] = 2
combined_data[nii_2.get_fdata()>0.5] = 3
combined_data[nii_3.get_fdata()>0.5] = 4
combined_data[nii_6.get_fdata()>0.5] = 6
combined_data[nii_9.get_fdata()>0.5] = 9

# save as one Nifti file
nib.save(nib.Nifti1Image(combined_data, nii_6.affine), 'combined_13_regions_v3.nii.gz')

