# Icecore_GMM_U-net
With the recent advancement of technology, it is now possible to have micro CT images of ice cores. These ice cores were collected from Greenland and Antarctica and brought to the AWI computer tomography lab for further analysis. The first step is to segment the images (binary segmentation) and pass through issues such as beam hardening, artifacts, and noise. As it is time-consuming to perform high-resolution images, I have produced binary masks from several high-resolution images with the GMM model, then downsized every two of them to one low-resolution mask (ground truth). Then these low-resolution masks were used as ground truth and low-resolution scans as model inputs.

High_res image ----> High_res mask with GMM ----> Downsizing to low-res mask ----> ground truth ----> U-net

# My GitHub Repository

Welcome to my GitHub repository! Here's an embedded 3D model:

<div class="sketchfab-embed-wrapper">
    <iframe title="Micro CT Polar Ice Bubbles" frameborder="0" allowfullscreen mozallowfullscreen="true" webkitallowfullscreen="true" allow="autoplay; fullscreen; xr-spatial-tracking" xr-spatial-tracking execution-while-out-of-viewport execution-while-not-rendered web-share src="https://sketchfab.com/models/0143f225daa34a5e8ba50987c288474d/embed"></iframe>
    <p style="font-size: 13px; font-weight: normal; margin: 5px; color: #4A4A4A;">
        <a href="https://sketchfab.com/3d-models/micro-ct-polar-ice-bubbles-0143f225daa34a5e8ba50987c288474d?utm_medium=embed&utm_campaign=share-popup&utm_content=0143f225daa34a5e8ba50987c288474d" target="_blank" rel="nofollow" style="font-weight: bold; color: #1CAAD9;">Micro CT Polar Ice Bubbles</a>
        by <a href="https://sketchfab.com/faramarz.bagherzadeh?utm_medium=embed&utm_campaign=share-popup&utm_content=0143f225daa34a5e8ba50987c288474d" target="_blank" rel="nofollow" style="font-weight: bold; color: #1CAAD9;">faramarz.bagherzadeh</a>
        on <a href="https://sketchfab.com?utm_medium=embed&utm_campaign=share-popup&utm_content=0143f225daa34a5e8ba50987c288474d" target="_blank" rel="nofollow" style="font-weight: bold; color: #1CAAD9;">Sketchfab</a>
    </p>
</div>

To interact with the 3D model, you can also [view it directly on Sketchfab](https://sketchfab.com/3d-models/micro-ct-polar-ice-bubbles-0143f225daa34a5e8ba50987c288474d).
