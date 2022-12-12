import cv2 
import numpy as np
import matplotlib.pyplot as plt
import statistics
from PIL import Image
kid_degraded=cv2.imread('./Kid2_degraded.tiff',cv2.IMREAD_GRAYSCALE)

sub_kid_degraded=kid_degraded[:100,:100]
print(sub_kid_degraded.shape)
number255=0
number0=0
num=[0]*254
name_list=[]
for a in range(256):
    name_list.append(f'{a}')
    
for i in range(100):
    for j in range(100):
        if sub_kid_degraded[i][j]==255:
            number255+=1
        if sub_kid_degraded[i][j]==0:
            number0+=1
num.insert(0,float(number0/10000))
num.append(float(number255/10000))
plt.figure(figsize=(30,30))
plt.bar(range(256),num,)
plt.tick_params(axis='x', labelsize=6)
plt.xticks(range(256), name_list,rotation='vertical')
plt.show()

kid_degraded_padding=cv2.copyMakeBorder(kid_degraded, 2,2,2,2, cv2.BORDER_CONSTANT, value=0)

def QuickSort(array):
    less = []
    equal = []
    greater = []

    if len(array) > 1:
        pivot = array[0]
        for x in array:
            if x < pivot:
                less.append(x)
            elif x == pivot:
                equal.append(x)
            elif x > pivot:
                greater.append(x)

        return QuickSort(less)+equal+QuickSort(greater)  
    else:  
        return array

def ftAlphaTrimmedMean(image):
    img = image.copy()
    arr=[]
    new_img=[]
    for i in range(800):
        for j in range(800):
            arr=[]
            for u in range(-2,3):
                for v in range(-2,3):
                    arr.append(img[i+2+u][j+2+v])
            arr=np.array(arr)
            arr=QuickSort(arr)
            del_arr=arr[8:17]
            new_img.append(statistics.mean(del_arr))
    return np.array(new_img).reshape(800,800)
delnoise_img=ftAlphaTrimmedMean(kid_degraded_padding)
plt.figure(figsize=(5,5))
plt.imshow(delnoise_img,cmap='gray')
plt.axis('off',)
plt.show()

kid_DFT = np.fft.fft2(delnoise_img) 
kid_DFTShift = np.fft.fftshift(kid_DFT)  
M,N = kid_DFT.shape
H = np.zeros((M,N), dtype=np.float32)
D0 = 250 # cut of frequency
n = 4 # order 
for u in range(M):
    for v in range(N):
        D = np.sqrt((u-M/2)**2 + (v-N/2)**2)
        H[u,v] = 1 / (1 + (D/D0)**n)



GLPF = np.zeros((800,800), dtype=np.float32)
D0 = 200
for u in range(800):
    for v in range(800):
        D = np.sqrt((u-800/2)**2 + (v-800/2)**2)
        GLPF[u,v] = np.exp(-D**2/(2*D0*D0))

Gshift = kid_DFTShift * H /GLPF
kid_invshift = np.fft.ifftshift(Gshift)  
kid_invDFT = np.abs(np.fft.ifft2(kid_invshift))
KID_LPF=np.uint8(cv2.normalize(kid_invDFT, None, 0, 255, cv2.NORM_MINMAX))
KID_LPF=Image.fromarray(KID_LPF)
# KID_LPF.save('KID_LPF.tif',dpi=(200.0,200.0))
plt.figure(figsize=(5,5))
plt.imshow(kid_invDFT, cmap='gray')
plt.axis('off')
plt.show()