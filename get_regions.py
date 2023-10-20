from function_for_combine_mask import find_and_read_nifti_data, process_13_regions_mask
import nibabel as nib
import os
import subprocess


def regions_generation(input_folder, save_seg, output_path):

    # set the command-line arguments as needed.
    command = (f'python ./totalsegmentator/TotalSegmentator.py -i "{input_folder}" -o "{save_seg}" '
               # f'--output_type "{out_type}" '
               f'--roi_subset heart small_bowel lung_lower_lobe_right lung_lower_lobe_left lung_middle_lobe_right '
               f'stomach pancreas duodenum kidney_right aorta inferior_vena_cava portal_vein_and_splenic_vein '
               f'urinary_bladder kidney_left iliac_artery_left iliac_artery_right iliac_vena_right iliac_vena_left '
               f'sacrum hip_left hip_right liver colon gallbladder spleen sacrum')

    # Run the command
    subprocess.run(command, shell=True)

    print("Process for segmentator is complete, start combine masks for 13 regions.")
    print("Start combining masks for 13 regions.")

    found_data = find_and_read_nifti_data(save_seg)
    print("Finish loading organ data.")
    liver_seg = nib.load(os.path.join(save_seg, "liver.nii.gz"))
    affine = liver_seg.affine
    # List the front and background masks for each region
    r0_front = ["colon.nii.gz"]
    r1_front = ["gallbladder.nii.gz", "liver.nii.gz"]
    r2_front = ["stomach.nii.gz", "pancreas.nii.gz", "liver.nii.gz"]
    r3_front = ["spleen.nii.gz"]
    # r6_front = ["urinary_bladder.nii.gz", "colon.nii.gz"]
    r6_front = ["urinary_bladder.nii.gz"]
    r9_front = ["small_bowel.nii.gz"]

    r0_bg = [
        "liver.nii.gz",
        "stomach.nii.gz",
        "kidney_right.nii.gz",
        "kidney_left.nii.gz",
        "small_bowel.nii.gz",
        "urinary_bladder.nii.gz",
        "spleen.nii.gz",
        "gallbladder.nii.gz",
        "pancreas.nii.gz",
        "aorta.nii.gz",
        "inferior_vena_cava.nii.gz",
        "portal_vein_and_splenic_vein.nii.gz",
    ]
    r1_bg = [
        "colon.nii.gz",
        "small_bowel.nii.gz",
        "stomach.nii.gz",
        "pancreas.nii.gz",
        "kidney_right.nii.gz",
        "aorta.nii.gz",
        "inferior_vena_cava.nii.gz",
        "portal_vein_and_splenic_vein.nii.gz"
    ]
    r2_bg = [
        "colon.nii.gz",
        "small_bowel.nii.gz",
        "spleen.nii.gz",
        "kidney_left.nii.gz",
        "aorta.nii.gz",
        "inferior_vena_cava.nii.gz",
        "portal_vein_and_splenic_vein.nii.gz"
    ]
    r3_bg = [
        "stomach.nii.gz",
        "kidney_left.nii.gz",
        "aorta.nii.gz",
        "inferior_vena_cava.nii.gz",
        "portal_vein_and_splenic_vein.nii.gz",
        "colon.nii.gz",
        "small_bowel.nii.gz"
    ]

    r6_bg = [
        "small_bowel.nii.gz",
        "aorta.nii.gz",
        "inferior_vena_cava.nii.gz",
        # "liver.nii.gz",
        # "stomach.nii.gz",
        # "kidney_right.nii.gz",
        # "kidney_left.nii.gz",
        # "small_bowel.nii.gz",
        # "spleen.nii.gz",
        # "gallbladder.nii.gz",
        # "pancreas.nii.gz",
        # "portal_vein_and_splenic_vein.nii.gz",
        # "iliac_artery_left.nii.gz",
        # "iliac_artery_right.nii.gz",
        # "iliac_vena_left.nii.gz",
        # "iliac_vena_right.nii.gz"
    ]
    r9_bg = [
        "urinary_bladder.nii.gz",
        "colon.nii.gz",
        "stomach.nii.gz",
        "gallbladder.nii.gz",
        "spleen.nii.gz",
        "liver.nii.gz",
        "kidney_right.nii.gz",
        "kidney_left.nii.gz",
        "pancreas.nii.gz",
        "aorta.nii.gz",
        "inferior_vena_cava.nii.gz",
        "portal_vein_and_splenic_vein.nii.gz"
    ]

    # Created the output folder before save the nifti files.
    save_dir = os.path.join(output_path, '13_regions')
    os.makedirs(save_dir, exist_ok=True)
    r0_seg = process_13_regions_mask(r0_front, r0_bg, found_data,
                                     affine, os.path.join(save_dir, 'region_0.nii.gz'), 0, save_seg)
    print("mask for Region 0 is complete.")
    _ = process_13_regions_mask(r1_front, r1_bg, found_data,
                                affine,  os.path.join(save_dir, 'region_1.nii.gz'), 1, save_seg)
    print("mask for Region 1 is complete.")
    _ = process_13_regions_mask(r2_front, r2_bg, found_data,
                                affine,  os.path.join(save_dir, 'region_2.nii.gz'), 2, save_seg)
    print("mask for Region 2 is complete.")
    _ = process_13_regions_mask(r3_front, r3_bg, found_data,
                                affine, os.path.join(save_dir, 'region_3.nii.gz'), 3, save_seg)
    print("mask for Region 3 is complete.")

    _ = process_13_regions_mask(r6_front, r6_bg, found_data,
                                affine, os.path.join(save_dir, 'region_6.nii.gz'), 6, save_seg, region_6=True)
    print("mask for Region 6 is complete.")
    r9_seg = process_13_regions_mask(r9_front, r9_bg, found_data,
                                     affine, os.path.join(save_dir, 'region_9.nii.gz'), 9, save_seg)
    print("mask for Region 9 is complete.")

    # Define the regions that same as r0 and r9
    r0_region = [4, 5, 7, 8]
    r9_region = [10, 11, 12]

    for region in r0_region:
        filename = os.path.join(output_path, f'13_regions/region_{region}.nii.gz')
        nib.save(r0_seg, filename)
        print(f"mask for Region {region} is complete.")

    for region in r9_region:
        filename = os.path.join(output_path, f"13_regions/region_{region}.nii.gz")
        nib.save(r9_seg, filename)
        print(f"mask for Region {region} is complete.")
