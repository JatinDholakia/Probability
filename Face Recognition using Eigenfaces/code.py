import cv2
import numpy as np

img = np.zeros((10304,320),dtype = np.uint8) # Creating zero matrix to store all the images. Dimensions = (size of flattened image, total training images)
for i in range(1,41):
    for j in range(1,9):
        I = cv2.imread('Images/s'+str(i)+'/'+str(j)+'.pgm',0) # Reading image to I
        I = np.ndarray.flatten(I)           # Flattened image stored to I
        img[:,8*(i-1)+(j-1)] = I            # Flattened image stored as a column of matrix img

mean = np.mean(img,axis=1)                  # Taking row wise mean of all images
mean = np.reshape(mean,(10304,1))           # Reshaping the mean as a vector
A = img-mean                                # Normailsing all the training images by subtracting mean. Dimension = (10304,320)

Cov = np.matmul(A,np.transpose(A))          # Obtaining the covariance matrix by the operation (A)*(At). Dimension = (10304,10304)
At_A = np.matmul(np.transpose(A),A)         # Obtaining the matrix (At)*A to calculate the eigenvalue and eigenvectors of the Covariance matrix. Dimension = (320,320)
u,v = np.linalg.eig(At_A)                   # Obtaining the eigenvalues(u) and eigenvectors(v) of (At)*A
ui = np.matmul(A,v)                         # Eigenvalue of (At)*A is same as eigenvalue of Covariance matrix. Eigenvectors of Covariance matrix is given by (A)*(v)
K = 52                                      # K is the number of most significant eigenvectors chosen. It is obtained by plotting the number of errors vs K
indices = np.argsort(-u)[:K]                # Sorting the eigenvalues and obtaining the indices of K maximum eigenvalues
uk = ui[:,indices]                          # Obtaining the corresponding K eigenvectors. Dimension = (10304,K)
norm = np.linalg.norm(uk,axis=0)            # Finding column wise norm for all the eigenvectors
uk_norm = uk/norm                           # Dividing by the norm to normalize the eigenvectors
omega = np.matmul(np.transpose(uk_norm),A)  # Obtaining omega matrix. Dimension = (K,320)

Folder = input("Enter the Folder Number (1-40) : ") # Taking input for folder
Image = input("Enter the test image number (9 or 10) : ") # Taking input for image

theta_e = 3230                               # Threshold value chosen to check if a face known or unkown
def Test(Folder,Image):                      # Function which returns the predicted class of a test image. Input: Folder between 1-40. Image number between 9 and 10.
    unknown = cv2.imread('Images/s'+str(Folder)+'/'+str(Image)+'.pgm',0) # Reading the image from folder and image
    row,col = np.shape(unknown)
    if (row != 112 and col != 92):
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') # Creating instance for Haar Cascade
        face = face_cascade.detectMultiScale(unknown,1.05,3)                        # Detecting the coordinates of faces in the image
        if(len(face)==0):
            return "No face Present"                                                # If length of coordinates is 0, return "no face Detected"
        [x,y,w,h] = face[0]                                                         # Unpack the values of face
        unknown = unknown[y:y+h,x:x+w]                                              # Crop the image
        unknown = cv2.resize(unknown,(112,92))                                      # Resize it to the size of training images
    unknown = np.ndarray.flatten(unknown)                   # Flattening the test image
    unknown = np.reshape(unknown,(10304,1))                 # Reshaping the image to a column vector
    unknown = unknown-mean                                  # Subtracting mean
    omega_test = np.matmul(np.transpose(uk_norm),unknown)   # Obtaining omega matrix for test image. It is given by transpose(eigenvectors)*test image
    distance = omega-omega_test                             # Obtaining the difference between omega and omega-test
    distance = np.linalg.norm(distance,axis=0)              # Finding the euclidean distance between omega and omega-test
    Test_index = np.argmin(distance)                        # The index of image that has the closest distance to test image is given stored in Test-index
    if (np.min(distance)<theta_e):
        return (Test_index/8) + 1                               # There are 8 training images in each class and index starts from 0.
    else:
        return "Unknown face"

print "Predicted Class:", Test(Folder,Image)
