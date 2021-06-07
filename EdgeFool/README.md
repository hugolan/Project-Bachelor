# EdgeFool
This is a modified copy of the work below, I use it in the context of my bachelor project to have grasp on the concept of adversarial images generation \n:
The official repository is [EDGEFOOL: AN ADVERSARIAL IMAGE ENHANCEMENT FILTER](https://arxiv.org/pdf/1910.12227.pdf)


### Image Smoothing 

Image smoothing is performed with the Python implementation of [Image Smoothing via L0 Gradient Minimization](http://www.cse.cuhk.edu.hk/~leojia/papers/L0smooth_Siggraph_Asia2011.pdf) provided by [Kevin Zhang](https://github.com/kjzhang/kzhang-cs205-l0-smoothing), as follows: 

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

A Fully Convolutional Neural Network (FCNN) is first trained end-to-end with a multi-task loss function which includes smoothing and adversarial losses. The architecture of the FCNN is instantiated from [Fast Image Processing with Fully-Convolutional Networks](https://arxiv.org/pdf/1709.00643.pdf) implemented in PyTorch by [Wu Huikai](https://github.com/wuhuikai/DeepGuidedFilter/tree/master/ImageProcessing/DeepGuidedFilteringNetwork). We enhance the image details of the L image channel only, after conversion to the Lab colour space without changing the colours of the image. In order to do this, we provided a differentiable PyTorch implementation of RGB-to-Lab and Lab-to-RGB. The enhanced adversarial images are then generated


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
* [Ali Shahin Shamsabadi](mailto:a.shahinshamsabadi@qmul.ac.uk)
* [Changjae Oh](mailto:c.oh@qmul.ac.uk)
* [Andrea Cavallaro](mailto:a.cavallaro@qmul.ac.uk)


## References
If you use our code, please cite the following paper:

      @InProceedings{shamsabadi2020edgefool,
        title = {EdgeFool: An Adversarial Image Enhancement Filter},
        author = {Shamsabadi, Ali Shahin and Oh, Changjae and Cavallaro, Andrea},
        booktitle = {Proceedings of the 45th IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP)},
        year = {2020},
        address = {Barcelona, Spain},
        month = May
      }
## License
The content of this project itself is licensed under the [Creative Commons Non-Commercial (CC BY-NC)](https://creativecommons.org/licenses/by-nc/2.0/uk/legalcode).
