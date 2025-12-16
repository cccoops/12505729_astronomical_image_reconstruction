# 12505729_astronomical_image_reconstruction

Python version: 3.11.2 <br>
This model uses a UNet architecture for reconstruction of blurry images. The final model can be found in the file 'best_cnn_unet.ipynb'. <br> 
Dataset: ESA Hubble Images - 3 classes (https://www.kaggle.com/datasets/subhamshome/esa-hubble-images-3-classes?select=galaxies). <br>
Note: to run model download images from the link, make a folder 'images' in directory, and add pictures to folder 'images' .
## Summary of model
The end-to-end pipeline goes as follows: 
- Load images from directory
- Process and cache images using TransformAndCacheDataset
- Images are loaded into data loaders split into 80% training and 20% test
- Model is initilised with the following hyperparameters: 
    - 50 epochs
    - Batch size of 8
    - Adam optimiser with a learning rate of 1e-4
- After training the models performance is evaluated

## Time spent on each task
- Retrieveing dataset and making a dataloader: 2 hours
- Setting up optimiser loss function: 2 hours
- Making the CNN UNet model: 24 hours
- Finding metrics: 1 hour
- Testing model: 16 hours
- Coming up with experiments: 8 hours
- Running experiments: 36 hours
- Documentation and README: 16 hours
- Other stuff: 5 hours 

## Evaluation of model
The model performace is evaluated by Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index Measure (SSIM). Both are measures great for determining the overall quality of an image compared to the original image where PSNR focuses on absolute pixel errors, while SSIM focuses on perceived structural changes. During the process, I have also manually compared the produced images with their original quality to determine the performance. Finally, I have also timed the model to weigh performace against speed. 

My goal was to reach a PSNR of $>32$ db and a SSIM of $0.9-1.0$. My goal was also to visually be conviced that the image is structurally similar to the original and that it contains the accurate features of astronomical phenomenon. 


## Experiments and results
Different experiments were run to test to achieve the best results.
I used a MacBook pro with a M1 chip, training for 50 epcos each experiment. As a baseline the average PSNR and SSIM for the blurry input images are: 
- PSNR: 31.67 dB
- SSIM: 0.8519

### Best model results
Optimiser: Adam (learning rate 1e-4)<br>
#batches: 8 <br>
Loss function: Charbonnier loss <br>
Results: <br>
- PSNR = 0.8955
- SSIM = 34.42 dB
- Failed to be visually convinced. 
- Time: 77.77 minutes

### Baseline model results
Optimiser: SGD (learning rate 1e-1)<br>
#batches: 8 <br>
Loss function: MSE loss <br>
Results: <br>
- PSNR = 0.8108 
- SSIM = 29.23 dB
- Failed to be visually convinced. 
- Time: 75.40 minutes

Experiments of only Adam and MSE are excluded. 