import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage
from matplotlib import cm

def displayImages(img1,img2,title1,title2):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(7,7))
    
    ax = fig.add_subplot(1, 2, 1)
    imgplot = plt.imshow(cv2.cvtColor(img1,cv2.COLOR_BGR2RGB))
    ax.set_title(title1)
    
    ax = fig.add_subplot(1, 2, 2)
    imgplot = plt.imshow(cv2.cvtColor(img2,cv2.COLOR_BGR2RGB))
    ax.set_title(title2)
    
    fig.tight_layout(pad=1.0)

    
# For Task 2.2
def displayImage(img,title):
    plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()

def rgbExclusion(img, channel):
    if channel == 'R': 
        img[:,:,2] = 0 #empty red channel
        return img
    elif channel == 'G': 
        img[:,:,1] = 0 #empty green channel
        return img
    elif channel == 'B': 
        img[:,:,0] = 0 #empty blue channel
        return img

    
# For Task 2.3
def displayHistogramEqualization(img,equImg):
    fig = plt.figure(figsize=(8,8))

    ax = fig.add_subplot(2, 2, 1)
    imgplot = plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    ax.set_title('Grayscale Image')
    
    ax = fig.add_subplot(2, 2, 2)
    plt.hist(img.ravel(),256,[0,256])
    ax.set_title('Histogram')
    
    ax = fig.add_subplot(2, 2, 3)
    imgplot = plt.imshow(cv2.cvtColor(equImg,cv2.COLOR_BGR2RGB))
    ax.set_title('Equalized Image')
    
    ax = fig.add_subplot(2, 2, 4)
    plt.hist(equImg.ravel(),256,[0,256])
    ax.set_title('Equalized Histogram')
    
    fig.tight_layout(pad=1.0)    
    imgplot = plt.show()
    
    
# For Task 2.4
def myConvolve2d(image, kernel):
    kernel = np.flipud(np.fliplr(kernel))    # Flip the kernel
    output = np.zeros_like(image)            # convolution output
    # Add zero padding to the input image
    image_padded = np.zeros((image.shape[0] + 2, image.shape[1] + 2))   
    image_padded[1:-1, 1:-1] = image
    
    kernal_x = kernel.shape[0]
    kernal_y = kernel.shape[1]
    
    for x in range(image.shape[0]):     # Loop over every pixel of the image
        for y in range(image.shape[1]):
            output[x,y]=(kernel*image_padded[x:x+kernal_x,y:y+kernal_y]).sum()
        
    return output


# For Task 2.5
def AddNoise(img, mode):
    if mode is not None:
        return skimage.util.random_noise(img, mode=mode)

def displayThreeImages(img1,img2,img3,title1,title2,title3):
    fig = plt.figure(figsize=(8,8))
    
    ax = fig.add_subplot(2, 3, 1)
    imgplot = plt.imshow(img1)
    ax.set_title(title1)
    
    ax = fig.add_subplot(2, 3, 2)
    imgplot = plt.imshow(img2)
    ax.set_title(title2)
    
    ax = fig.add_subplot(2, 3, 3)
    imgplot = plt.imshow(img3)
    ax.set_title(title3)
    
    fig.tight_layout(pad=1.0)

def drawMeshPlot(x,y,z1,z2,z3):
    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(131, projection='3d')
    ax.plot_surface(x,y,z1, cmap=cm.jet)
    ax.set_title("sigma=2")

    ax = fig.add_subplot(132, projection='3d')
    ax.plot_surface(x,y,z2, cmap=cm.jet)
    ax.set_title("sigma=2.5")

    ax = fig.add_subplot(133, projection='3d')
    ax.plot_surface(x,y,z3, cmap=cm.jet)
    ax.set_title("sigma=4")
    plt.show()
    
    
# For Task 2.6
def displayFourImages(img1,img2,img3,img4,title1,title2,title3,title4):
    fig = plt.figure(figsize=(12,12))
    
    ax = fig.add_subplot(1, 4, 1)
    imgplot = plt.imshow(cv2.cvtColor(img1,cv2.COLOR_BGR2RGB))
    ax.set_title(title1)
    
    ax = fig.add_subplot(1, 4, 2)
    imgplot = plt.imshow(cv2.cvtColor(img2,cv2.COLOR_BGR2RGB))
    ax.set_title(title2)
    
    ax = fig.add_subplot(1, 4, 3)
    imgplot = plt.imshow(cv2.cvtColor(img3,cv2.COLOR_BGR2RGB))
    ax.set_title(title3)
    
    ax = fig.add_subplot(1, 4, 4)
    imgplot = plt.imshow(cv2.cvtColor(img4,cv2.COLOR_BGR2RGB))
    ax.set_title(title4)
    
    fig.tight_layout(pad=1.0)
    
    def meshPlot(x,y,z):
    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x,y,zz, cmap=cm.jet)
    ax.set_title("sigma=1")
    plt.show()