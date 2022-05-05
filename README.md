# Palmprint-Recognition
Palmprint Recognition with Neural Networks implemented with PyTorch.
For training of the networks, I can recommend using Kaggle, as you have also a GPU Accelerator available there, if you are also running solely on CPU like I do. :-)

# Installation and Usage
In order to run all notebooks and also to test the prototype you need to install the
environment.yml file placed in the Prototype folder.

If you want to train any of the provided networks in the jupyter notebooks, you need to get access to the Tongji palmprint and NTU Palmprints from the Internet (NTU-PI-v1)[[1]](#1) datasets and do some image preprocessing and structuring first. Therefore the files in the utils folder might be helpful. Also there is one file called 'FeatureCreator' where you can produce normalized theta vectors for the Spatial Transformer Network training by yourself! In my models I used it to also train on the raw Tongji images.

# Classification with CNN
For classification of the ROIs classic network architectures like ResNet18, VGG and others are trained based on [[2]](#2). All hyperparamters were set like in the mentioned paper, which gave very promising results in the end.

# Spatital Transformer Network
The PyTorch implementation of Spatial Transformer Network (STN) with Thin Plate Spline (TPS) is used from https://github.com/WarBean/tps_stn_pytorch. Thank you very much WarBean!

The main idea and structure of the ROI extractor is based on the ROILaNet developed by [[1]](#1)
![image](https://user-images.githubusercontent.com/48522299/166965447-6ec02878-5176-4782-87da-06b3bfdcea4e.png)

# Performance
All trained models and notebooks for model training can be found in the Model-Training folder.
## Performance of the classifier, on self created Tongji ROI images
<img src="https://user-images.githubusercontent.com/48522299/166965920-dce7ff97-bfe6-437e-8f23-5967dbaa69b2.JPG" width="400" height="300">
<img src="https://user-images.githubusercontent.com/48522299/166965971-63bdbb70-86bc-44fd-8545-3f9653bcd347.JPG" width="400" height="300">

## Performance of the rebuilt ROILaNet
![loss](https://user-images.githubusercontent.com/48522299/166966083-e0d1719e-ae7b-42a6-867c-9e0755d2600f.png)

# User Interface
I also created a small and easy-to-use GUI, where you can select Tongji images (I provided some in the data-folder inside the Prototype-folder) and check the prediction steps my so called Palmprint Recognition Network (PRN) is performing.

![image](https://user-images.githubusercontent.com/48522299/166967167-09d183f1-49f2-46ed-bee8-d1208678d5bb.png)


# Literature List
<a id="1">[1]</a> Wojciech Michal Matkowski, Tingting Chai and Adams Wai Kin Kong. “Palmprint Recognition in Uncontrolled and Uncooperative Environment.” IEEE Transactions on Information Forensics and Security, October 2019, DOI: 10.1109/TIFS.2019.2945183.

<a id="2">[2]</a> Jia, W., Gao, J., Xia, W. et al. A Performance Evaluation of Classic Convolutional Neural Networks for 2D and 3D Palmprint and Palm Vein Recognition. Int. J. Autom. Comput. 18, 18–44 (2021). https://doi.org/10.1007/s11633-020-1257-9
# Credits
If you want to use an of my individual code in your work, link my github and also mention me:
```yaml
@misc{holzweberPalm2022,
  author = {Christopher Holzweber},
  title = {Palmprint Recognition with Neural Networks},
  year = {2022},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/holzweber/palmprint-recognition/}}
}
