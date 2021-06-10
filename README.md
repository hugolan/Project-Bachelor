# Project-Bachelor

# EdgeFool
This is a modified copy of the original repositery, I use it in the context of my bachelor project to have grasp on the concept of adversarial images generation:
This code works as follows: 
   
   - First generate, the smoothed images 
   - Secondly, enhanced the details and add the imperceptible noise
   
### Image Smoothing 

1. Go to Smoothing directory
   ```
   cd Smoothing
   ```
2. Smooth the original images
   ```
   bash script.sh
   ```
3. The l0 smoothed images will be saved in the SmoothImgs directory (within the 'root' directory) with the same name as their corresponding original images

### Generate the enhanced adversarial images

1. Go to Train directory
   ```
   cd Train
   ```
2. In the script.sh set the paths of
(i) directory of the original images,
(ii) directory of the smoothed images, and
(iii) classifier under attack. The current implementation supports three classifiers Resnet18, Resnet50 and Alexnet, however other classifiers can be employed by changing the lines (80-88) in train_base.py.
3. Generate the enhanced adversarial images 
   ```
   bash script.sh
   ```
4. The enhanced adversarial images are saved in the EnhancedAdvImgsfor_{classifier} (within the 'root' directory) with the same name as their corresponding original images


### Generate the enhanced adversarial images
Obtain results from adversarial Images

1. Go to Train directory
   ```
   cd Train
   ```
2. Launch Test from already computed model
   ```
   bash exec.sh
   ```
### To get the missleading rate:

# ColorFool

## Description
The code works in two steps: 
1. Identify image regions using semantic segmentation model
2. Generate adversarial images via perturbing color of semantic regions in the natural color range    

### Semantic Segmentation 

1. Go to Segmentation directory
   ```
   cd Segmentation
   ```
2. Download segmentation model (both encoder and decoder) from [here](https://drive.google.com/drive/folders/1FjZTweIsWWgxhXkzKHyIzEgBO5VTCe68) and locate in "models" directory.
   

3. Run the segmentation for all images within Dataset directory (requires GPU)
   ```
   bash script.sh
   ```

The semantic regions of four categories will be saved in the Segmentation/SegmentationResults/$Dataset/ directory as a smooth mask the same size of the image with the same name as their corresponding original images

### Generate ColorFool Adversarial Images

1. Go to Adversarial directory
   ```
   cd ../Adversarial
   ```
2. In the script.sh set 
(i) the name of target models for attack, and (ii) the name of the dataset.
The current implementation supports three classifiers (Resnet18, Resnet50 and Alexnet) trained with ImageNet.
3. Run ColorFool for all images within the Dataset directory (works in both GPU and CPU)
   ```
   bash script.sh
   ```

### Outputs
* Adversarial Images saved with the same name as the clean images in Adversarial/Results/ColorFoolImgs directory;
* Metadata with the following structure: filename, number of trials, predicted class of the clean image with its probablity and predicted class of the adversarial image with its probablity in Adversarial/Results/Logs directory.


## Setup

To be able to use this code you need to follow this instructions:

1. Download source code from GitHub
   ```
    git clone https://github.com/hugolan/Project-Bachelor.git 
   ```
2. Create [conda](https://docs.conda.io/en/latest/miniconda.html) virtual-environment
   ```
    conda create --name Environment python=3.5.6
    (For edge use: conda create Environment --name python=2.7.15)
   ```
3. Activate conda environment
   ```
    source activate Environment
   ```
4. Install requirements for EdgeFool and ColorFool
   ```
    pip install -r requirements.txt
   ```
 N.B: First install the requirements before trying to use the visualization code, otherwise it may not work without conda.
 
# CAM

Use the following command to obtain the result from python: 
   ```
       python pytorch_CAM.py
   ```

Name cam_visualize_with_python the image you want to be analyzed. But it can be changed in the file at line 93 using OpenCV.

# Pytorch-CNN


## Authors
EdgeFool:

* [Ali Shahin Shamsabadi](mailto:a.shahinshamsabadi@qmul.ac.uk)
* [Changjae Oh](mailto:c.oh@qmul.ac.uk)
* [Andrea Cavallaro](mailto:a.cavallaro@qmul.ac.uk)

ColorFool:

* [Ali Shahin Shamsabadi](mailto:a.shahinshamsabadi@qmul.ac.uk)
* [Ricardo Sanchez-Matilla](mailto:ricardo.sanchezmatilla@qmul.ac.uk)
* [Andrea Cavallaro](mailto:a.cavallaro@qmul.ac.uk)

CAM:

* [Bolei Zhou] https://github.com/zhoubolei

Pytorch-CNN:

* [Utku Ozbulak] https://github.com/utkuozbulak


## References

EdgeFool: https://arxiv.org/pdf/1910.12227.pdf

ColorFool: https://arxiv.org/pdf/1911.10891.pdf

Class Activation Mapping: https://github.com/zhoubolei/CAM

Pytorch-cnn visualization: https://github.com/utkuozbulak/pytorch-cnn-visualizations

## License
The content of this project itself is licensed under the [Creative Commons Non-Commercial (CC BY-NC)](https://creativecommons.org/licenses/by-nc/2.0/uk/legalcode).
