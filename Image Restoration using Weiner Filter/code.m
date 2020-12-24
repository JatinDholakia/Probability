clear all
% The builtin image names of MATLAB are stored in a cell array.
train_names = {'cameraman.tif','liftingbody.png','m83.tif','onion.png','peppers.png','saturn.png','tape.png','tire.tif','greens.jpg','shadow.tif'};
for i=1:10
    % Reading the image to double data type
    img = imread(train_names{i});
    if(ndims(img)==3)
        img = im2double(rgb2gray(img));
    end
    f(:,:,i) = im2double(imresize(img,[256 256]));
    
    %% Adding Noise to Image
    % Creating a Gaussian Point Spread Function of Size (5,5) and Standard
    % Deviation = 5 and blurring the image with imfilter.
    PSF = fspecial('gaussian',[5 5],5);
    blurred(:,:,i) = imfilter(f(:,:,i), PSF, 'conv', 'symmetric');
    
    % Additive White Gaussian Noise is added with mean=0 and Variance = 0.001.
    noise_mean = 0;
    noise_var = 0.01;
    g(:,:,i) = imnoise(blurred(:,:,i), 'gaussian',noise_mean, noise_var);
    
    %% Making Wiener Filter
    % Fourier transform of the Noisy image and Point Spread Function are calculated. They are made to be of same dimensions
    N = size(f(:,:,i),1);
    G(:,:,i) = fft2(g(:,:,i));
    H(:,:,i) = fft2(PSF,N,N);
    
    % The Power Spectral Density of Original Image is calcuated.
    Pf(:,:,i) = abs(fft2(f(:,:,i))).^2/N^2;
    % Wiener Filter is made. W is the frequency response of the filter.
    W(:,:,i) = (conj(H(:,:,i)))./(abs(H(:,:,i)).^2 + noise_var./(Pf(:,:,i)));
    % Fourier Transform of the noisy image is multiplied with the Filter.
    F_hat(:,:,i) = W(:,:,i).*G(:,:,i);
    % Inverse Fourier transform is calculated and f_hat is estimated which
    % minimises the Mean Squared Error
    f_hat(:,:,i) = abs(ifft2(F_hat(:,:,i)));
    % The PSNR of the estimated image is calculated with respect to the
    % original image.
    psnr_custom(i) = psnr(f_hat(:,:,i),f(:,:,i));
end
% The Wiener Filter is made by taking the average of all Wiener Filters
% created in the train set.
W = mean(W,3); % Taking average in dimension=3

% Showing a sample
subplot(131);imshow(f(:,:,1));title('Original');
subplot(132); imshow(g(:,:,1)); title('Noisy');
subplot(133);imshow(f_hat(:,:,1)); title('Filtered');
%% Testing on Other Images
% 5 Test Images are used( Inbuilt MATLAB images)
test_names = {'coins.png','rice.png','moon.tif','football.jpg' ,'canoe.tif'};
% They are read in 'double' format and stored in an array.
for i=1:5
    img = imread(test_names{i});
    if(ndims(img)==3)
        img = rgb2gray(img);
    end
    I(:,:,i) = im2double(imresize(img,[256 256]));
    % Defocus Blur and Additive Gaussian Noise are added
    noisy(:,:,i) = add_noise(I(:,:,i),0.01);
    % Wiener Filter is applied
    output(:,:,i) = WienerFilter(noisy(:,:,i),W);
    % PSNR is calculated
    test_psnr(:,:,i) = psnr(output(:,:,i),I(:,:,i));
    psnr_noisy(:,:,i) = psnr(noisy(:,:,i),I(:,:,i));
end
Avg_psnr = mean(test_psnr); % Average PSNR is calculated.
disp('Average PSNR = ') 
disp(Avg_psnr)
%% Functions
%Function to add noise to image, with variance as input
function noisy = add_noise(f,noise_var)
psf = fspecial('gaussian',[5 5],5); % Creating Point Spread Function
blurred = imfilter(f, psf, 'conv', 'symmetric'); % Blurring Image

noise_mean = 0; % Mean of additive noise set to 0
noisy = imnoise(blurred, 'gaussian',noise_mean, noise_var); % Adding Noise
end

% Function to apply Wiener Filter to noisy Image.
% Inputs = Noisy Image and Frequency Response of the Filter
function output = WienerFilter(g,W)
G = fft2(g); % Fourier transform of the input image
O = G.*W; % Multiplication with Wiener Filter's frequency response
output = abs(ifft2(O)); % Inverse Fourier transform
end