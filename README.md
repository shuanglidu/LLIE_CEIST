# Low-light Image Enhancement based on Coarse Estimation of Illumination and Scene Texture

LLIE_CEIST

Zhao Minghua1, Wen Yichun1，Du Shuangli1（*）, Hu Jing1, Shi Cheng1, Li Peng1

*Corresponding Author  
 1 Xi'an University of Technology, Xi'an, China

The manuscript is submitted to the Journal of Image and Graphics (JIG) (中国图象图形学报).

# Requirements：
1.Python 3.6.5  
2.Pytorch 1.2.0  
3.Cuda 10.0  
4.Torchvision 0.11.1  

# Dataset preparing:
	You can download the LOL dataset (https://daooshee.github.io/BMVC2018website/) and other datasets (including DICM, LIME, MEF, NPE, VV etc.) to test our method with the pretrained model.
	The pretrained model can be found at the file ./checkpoints/enlightening.
	
# Predict:
	Firstly, use the matlab code (min channel constraint map.m) to produce the minimum channel constraint map for low-light images.
	Secondly, put low-light images and their corresponding minimum channel constraint maps into ../test_dataset/testA and ../test_dataset/testC ( And you should keep whatever one image in ../test_dataset/testB and ../test_dataset/testD to make sure program can start.)
	Then run Python predict.py
