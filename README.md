# Icecore_GMM_U-net
With the recent advancement of technology, it is now possible to have micro CT images of ice cores. These ice cores were collected from Greenland and Antarctica and brought to the AWI computer tomography lab for further analysis. The first step is to segment the images (binary segmentation) and pass through issues such as beam hardening, artifacts, and noise. As it is time-consuming to perform high-resolution images, I have produced binary masks from several high-resolution images with the GMM model, then downsized every two of them to one low-resolution mask (ground truth). Then these low-resolution masks were used as ground truth and low-resolution scans as model inputs.

High_res image ----> High_res mask with GMM ----> Downsizing to low-res mask ----> ground truth ----> U-net

![alt text](https://github.com/Faramarz-bagherzadeh/Icecore_GMM_U-net/blob/main/model_output.png)
