import argparse
from pathlib import Path
from combine_mask_of_13_regions import find_and_read_nifti_data, process_13_regions_mask
import nibabel as nib
import subprocess


def get_args_parser():

    parser = argparse.ArgumentParser(description="Combine organs' masks into 13 regions.")

    parser.add_argument("-i", metavar="filepath", dest="input",
                        help="CT nifti image or folder of dicom slices",
                        type=lambda p: Path(p).absolute(), required=True)

    parser.add_argument("-o", metavar="directory", dest="output",
                        help="Output directory for segmentation masks",
                        type=lambda p: Path(p).absolute(), required=True)
    return parser


def main():

    parser = get_args_parser()
    args = parser.parse_args()
    data_input = Path(args.input)
    output_path = Path(args.output)

    # # set the command-line arguments as needed.
    # command = f'python ./totalsegmentator/bin/TotalSegmentator.py -i "{data_input}" -o "{data_input}"'
    #
    # # Run the command
    # subprocess.run(command, shell=True)

    found_data = find_and_read_nifti_data(output_path)
    liver_seg = nib.load(output_path / f"liver.nii.gz")
    affine = liver_seg.affine
    # List the front and background masks for each region
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

    (output_path / '13_regions').mkdir(parents=True, exist_ok=True)

    # note, the output folder needed to be created before save the nifti files.
    r0_seg = process_13_regions_mask(r0_front, r0_bg, found_data,
                                     affine, output_path / f"13_regions/region_0.nii.gz")
    print("mask for Region 0 is complete.")
    _ = process_13_regions_mask(r1_front, r1_bg, found_data,
                                affine,  output_path / f"13_regions/region_1.nii.gz")
    print("mask for Region 1 is complete.")
    _ = process_13_regions_mask(r2_front, r2_bg, found_data,
                                affine,  output_path / f"13_regions/region_2.nii.gz")
    print("mask for Region 2 is complete.")
    _ = process_13_regions_mask(r3_front, r3_bg, found_data,
                                affine, output_path / f"13_regions/region_3.nii.gz")
    print("mask for Region 3 is complete.")
    _ = process_13_regions_mask(r6_front, r6_bg, found_data,
                                affine, output_path / f"13_regions/region_6.nii.gz")
    print("mask for Region 6 is complete.")
    r9_seg = process_13_regions_mask(r9_front, r9_bg, found_data,
                                     affine, output_path / f"13_regions/region_9.nii.gz")
    print("mask for Region 9 is complete.")

    # Define the regions that same as r0 and r9
    r0_region = [4, 5, 7, 8]
    r9_region = [10, 11, 12]

    for region in r0_region:
        filename = output_path / f"13_regions/region_{region}.nii.gz"
        nib.save(r0_seg, filename)
        print(f"mask for Region {region} is complete.")

    for region in r9_region:
        filename = output_path / f"13_regions/region_{region}.nii.gz"
        nib.save(r9_seg, filename)
        print(f"mask for Region {region} is complete.")


if __name__ == "__main__":

    main()
