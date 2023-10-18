#!/usr/bin/env python
# import sys
from pathlib import Path
import argparse
import subprocess
import nibabel as nib
from Region_segmentation.totalsegmentator.libs import combine_masks


def main():
    """
    Combine binary labels into a binary file.

    Works with any number of label files.

    Usage:
    totalseg_combine_masks -i totalsegmentator_output_dir -o combined_mask.nii.gz -m lung
    """
    parser = argparse.ArgumentParser(description="Combine masks.",
                                     epilog="Written by Jakob Wasserthal. If you use this tool please cite https://pubs.rsna.org/doi/10.1148/ryai.230024")

    parser.add_argument("-i", metavar="directory", dest="mask_dir",
                        help="TotalSegmentator output directory containing all the masks", 
                        type=lambda p: Path(p).absolute(), required=True)

    parser.add_argument("-o", metavar="filepath", dest="output",
                        help="Output path for combined mask", 
                        type=lambda p: Path(p).absolute(), required=True)

    parser.add_argument("-m", "--masks", type=str, choices=["lung", "lung_left", "lung_right", 
                        "vertebrae", "ribs", "vertebrae_ribs", "heart", "pelvis", "body"],
                        help="The type of masks you want to combine", required=True)

    parser.add_argument("-t", "--nora_tag", type=str, help="tag in nora as mask. Pass nora project id as argument.",
                        default="None")

    args = parser.parse_args()

    combined_img = combine_masks(args.mask_dir, args.masks)
    nib.save(combined_img, args.output)

    if args.nora_tag != "None":
        subprocess.call(f"/opt/nora/src/node/nora -p {args.nora_tag} --add {args.output} --addtag mask", shell=True)


if __name__ == "__main__":
    main()
