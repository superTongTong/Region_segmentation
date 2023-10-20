# Region_segmentation

## Command line for run the python code and explanations:

**You can copy following command line to run the python code:**  

```
python .\region_segmentation.py -i "<input directory>" -o <output folder name>
```
> **Note:** Don't forget modify the **input directory** and **output folder name** before you run the command line.  
> Example of command line looks like:  
> ``python .\region_segmentation.py -i "./data/example_folder" -o processed_data``
### Command line explanations:
**-i "./input/folder/directory"**:   
please fill in the directory of the input folder in the Double quotes.  

**-o name_of_output_folder**:   
you can give a output folder name, the code will create the folder which you named,
and save all the output files under this output folder. The output folder will have same structure as input folder.
## Steps for set_up python code working environment:
1. Download git repo.
2. Install pycharm
3. Create virtual environment.
4. Extract git repo, install all the package. (pip install -r requirements.txt)
5. Install CUDA.
   1. Check if GPU driver and CUDA is enabled and accessible by PyTorch  
   ```
   import torch 
   torch.cuda.is_available()
   ```
   2. If no CODA installed, run:   
   ```
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```
   3. Run pytorch again, check if the CUDA is installed.
6. Install powerToys keep computer awake for python code running overnight. or setting manually.
7. Prepare small folder for testing everything. For 858 slices DICOM scan (00017->1), it takes 12 minutes.
