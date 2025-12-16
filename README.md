# 12505729_astronomical_image_reconstruction

This model uses a UNet architecture for reconstruction of blurry images. 
Dataset: ESA Hubble Images - 3 classes (https://www.kaggle.com/datasets/subhamshome/esa-hubble-images-3-classes?select=galaxies)

## Summary of model
The end-to-end pipeline goes as follows: 
- Load images into a CLASS_NAME to make them blurry
- Cache images using CLASS_NAME
- Images are loaded into data loaders split into 80% training and 20% test
- Model is initilised with the following hyperparameters: 
    - 75 epochs
    - Batch size of 8
    - Adam optimiser with a learning rate of 1e-4
- After training the models performance is evaluated

## Evaluation of model
The model performace is evaluated by Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index Measure (SSIM). Both are measures great for determining the overall quality of an image compared to the original image where PSNR focuses on absolute pixel errors, while SSIM focuses on perceived structural changes. During the process, I have also manually compared the produced images with their original quality to determine the performance. 

My goal was to reach a PSNR of $>30$ db and a SSIM of $0.9-1.0$. My goal was also to visually be conviced that the image is structurally similar to the original and that it contains the accurate features of astronomical phenomenon. 

I was able to reach the following results: <br>
- PSNR = 0.89
- SSIM = 0.34
- Failed to be visually convinced. 


## Time spent on each task

- 
-
-
-
-

