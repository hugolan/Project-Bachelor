# Project-Bachelor

# EdgeFool
This is a modified copy of the work below, I use it in the context of my bachelor project to have grasp on the concept of adversarial images generation \n:
The official repository is [EDGEFOOL: AN ADVERSARIAL IMAGE ENHANCEMENT FILTER](https://arxiv.org/pdf/1910.12227.pdf)


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

## Authors
EdgeFool:

* [Ali Shahin Shamsabadi](mailto:a.shahinshamsabadi@qmul.ac.uk)
* [Changjae Oh](mailto:c.oh@qmul.ac.uk)
* [Andrea Cavallaro](mailto:a.cavallaro@qmul.ac.uk)

ColorFool:










## References

EdgeFool: https://arxiv.org/pdf/1910.12227.pdf

ColorFool: https://arxiv.org/pdf/1911.10891.pdf

## License
The content of this project itself is licensed under the [Creative Commons Non-Commercial (CC BY-NC)](https://creativecommons.org/licenses/by-nc/2.0/uk/legalcode).
