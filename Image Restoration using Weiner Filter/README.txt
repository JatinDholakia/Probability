The code is a MATLAB file.

The inbuilt MATLAB demo images are used for training and testing.
10 images used for training and 5 for testing.

The training images are blurred and white noise is added.
They are filtered with a Weiner Filter, which is made by using their respective Power Spectral Densities.
The resultant Weiner Filter is the average of filters used for training.

5 Test images are corrupted with Blur and Gaussian Noise.
They are filtered with the obtained Weiner Filter.
The average PSNR value is calculated by taking the original image as reference.


 