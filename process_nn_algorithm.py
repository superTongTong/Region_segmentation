from nn_algorithm import closing_image, find_non_overlap, divide_overlapped_region
import time
import SimpleITK as sitk
import os
from tqdm import tqdm
import argparse
import slicerio
import nrrd
import glob
from nrrd_to_nifiti_conversion import nifti_write


def extract_3_regions(input_folder, output_folder, num_regions1, num_regions2):

    ''''
    following code is to extract the segmentations from the original segmentation file
    '''
    # Get a list of .nrrd files in a directory
    nrrd_files = glob.glob(input_folder)
    count = 1
    for file in tqdm(nrrd_files):

        seg_label = f's{file[-13:-9]}'
        print(f'Start processing scan {seg_label}....')
        # input_filename = os.path.join(input_path, file)
        # load the segmentation data information
        segmentation_info = slicerio.read_segmentation_info(file)
        segment_names = slicerio.segment_names(segmentation_info)

        # create an empty list to store the segment names and labels
        segment_names_to_labels = []

        # create list of name and labels for the 13 regions
        # example format: segment_names_to_labels = [("Segment_1_1", 1), ("Segment_1_1", 2), ("Segment_1_2", 3)]
        for i in range(num_regions1, num_regions2+1): # Currently we onty have 3 regions
            sge_name_label = (segment_names[i], i)
            segment_names_to_labels.append(sge_name_label)

        voxels_data, header = nrrd.read(file)

        # extract the 13 regions from the original segmentation file
        extracted_voxels, extracted_header = slicerio.extract_segments(voxels_data, header, segmentation_info,
                                                                       segment_names_to_labels)
        extracted_header['dimension'] = 3
        # directory = '../data/three_regions_segmentation_orig/masks_1_3'
        os.makedirs(output_folder, exist_ok=True)
        save_dir = os.path.join(output_folder, seg_label)

        nifti_write(extracted_voxels, extracted_header, prefix=save_dir)
        print(f'Finish processing scan {seg_label}, currently {count} files processed.')
        count += 1


def nearest_neighbour_process(folder_in, folder_out, nn_margin=3):
    # Check if the input folder exists
    if not os.path.exists(folder_in):
        raise FileNotFoundError(f"The input folder '{folder_in}' does not exist.")
    # Create the output folder if it doesn't exist
    os.makedirs(folder_out, exist_ok=True)

    ''''here call the function from nn_algorithm.py to process the data'''
    # List all files in the input folder
    nii_files = os.listdir(folder_in)

    for file in tqdm(nii_files):
        # if file.endswith('.nii'):
        suffixes = ('.nii', '.nii.gz')
        if file.endswith(suffixes):
            sitk_orig = sitk.ReadImage(f"{folder_in}/{file}", sitk.sitkInt8)
            array_orig = sitk.GetArrayFromImage(sitk_orig)
            c_image, t_image = closing_image(sitk_orig, kernel_radius=[3, 3, 3])
            non_overlapped_voxel = find_non_overlap(c_image, t_image)

            processed_image = divide_overlapped_region(non_overlapped_voxel, array_orig, nn_margin)

            # covert the processed image to sitk image
            img_for_save = sitk.GetImageFromArray(processed_image)
            img_for_save.CopyInformation(sitk_orig)
            save_dir = f"{folder_out}/after_nn"
            os.makedirs(save_dir, exist_ok=True)

            # Export Image
            sitk.WriteImage(img_for_save, f"{save_dir}/{file}")

def main(args):
    # input_path = 'data/nnunet/raw/Dataset018_Orig_nn/labelsTr'
    # output_path = 'data/nnunet/raw/Dataset018_Orig_nn/'
    save_dir = f'{args.output}extracted_regions/'
    extract_3_regions(args.input, save_dir, args.num_regions1, args.num_regions2)

    nearest_neighbour_process(save_dir, args.output, args.nn_margin)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input', default=r"data/nrrd/*.nrrd", type=str,
                        help='dicom input folder path (default: data/nrrd/*.nrrd)')
    parser.add_argument('-o', '--output',
                        default=r"data/nrrd/nn_output/",
                        type=str, help='NIfTI output folder path (default: data/nnunet/raw/Dataset018_Orig_nn/)')
    parser.add_argument('-nr1', '--num-regions1',
                        default=0,
                        type=int, help='start number of regions wanted to extract (default: 0)')
    parser.add_argument('-nr2', '--num-regions2',
                        default=4,
                        type=int, help='end number of regions wanted to extract(default: 4)')
    parser.add_argument('-nnm', '--nn-margin',
                        default=3,
                        type=int, help='Nearest neighbour margin (default: 3mm)')

    start_time = time.time()
    args = parser.parse_args()
    main(args)
    total_time = time.time() - start_time
    print(f"--- {total_time:.2f} seconds ---")