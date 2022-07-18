# 3d-verification
## Supplementary work
In the supplementary_work.pdf, we demonstrate the power of JANet and provide the network configuration.
## First build the enviroment:

conda create --name cnncert python=3.6
source activate cnncert
conda install pillow numpy scipy pandas h5py tensorflow numba posix_ipc matplotlib
The Tensorflow version should below tf1.15
## Then download our model checkpoints
https://livelancsac-my.sharepoint.com/:f:/g/personal/mur2_lancaster_ac_uk/Eirostdd_-tOjpZmkU-yFdkBF6auqdp3IgWDur3ZcTnkyg?e=3fUXhe
## To obtain the average bounds of 64 points on 12 layes with average pooling in PointNet model with T-Net, you can run 
python main.py
## To obtain the distortion from attack method, you could run
python atmain.py
