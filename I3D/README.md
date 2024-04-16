# From video to I3D feature
Code copied (and simplified) from https://github.com/RenzKa/sign-segmentation  

## Objective : given a video, extract I3D features easily 
### 1. Select checkpoints & save it in models/i3d/ 
https://github.com/RenzKa/sign-segmentation : i3d_kinetics_bsl1k_bslcp.pth.tar , i3d_kinetics_bslcp.pth.tar , i3d_kinetics_phoenix_1297.pth.tar (num_classes = 981)

https://www.robots.ox.ac.uk/~vgg/research/bslattend/ : bsl5k.pth.tar (num_classes = 5383) --> the one we choose.

### 2. Make I3D features
run `I3Dmakefeatures.py` (check you have the right number of classes)
