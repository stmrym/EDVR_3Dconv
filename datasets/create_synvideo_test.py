from __future__ import division
import os, cv2, sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
from tqdm import tqdm
import time

'''
create synthetic videos from REDS dataset.

[Input] REDS_dataset/train_sharp (please download from https://seungjunnah.github.io/Datasets/reds.html)

[Output] test_RR, test_GT, test_R_blur_mask

'''

time_sta = time.perf_counter()

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def gkern(kernlen=100, nsig=1):
    """Returns a 2D Gaussian kernel array."""
    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    kernel = kernel/kernel.max()
    return kernel



def syn_data(t,r,sigma, alpha1, alpha2):
    t=np.power(t,2.2)
    r=np.power(r,2.2)
    
    sz=int(2*np.ceil(2*sigma)+1)
    r_blur=cv2.GaussianBlur(r,(sz,sz),sigma,sigma,0)
    blend=r_blur+t

    for i in range(3):
        maski=blend[:,:,i]>1
        mean_i=max(1.,np.sum(blend[:,:,i]*maski)/(maski.sum()+1e-6))
        r_blur[:,:,i]=r_blur[:,:,i]-(mean_i-1)*const_att
    r_blur[r_blur>=1]=1
    r_blur[r_blur<=0]=0


    r_blur_mask=np.multiply(r_blur,alpha1)
    blend=r_blur_mask+t*alpha2
    
    t=np.power(t*alpha2,1/2.2)
    r_blur_mask=np.power(r_blur_mask,1/2.2)
    blend=np.power(blend,1/2.2)
    blend[blend>=1]=1
    blend[blend<=0]=0

    return t,r_blur_mask,r_blur, blend

def prepare_data(t_root_dir, r_root_dir):
    t_list=[]
    r_list=[]
    for root, _, fnames in sorted(os.walk(t_root_dir)):
        for fname in fnames:
            if is_image_file(fname):
                path_t = os.path.join(t_root_dir, fname)
                path_r = os.path.join(r_root_dir, fname)
                t_list.append(path_t)
                r_list.append(path_r)
                
    return t_list,r_list

############################################

# create test dataset from '230' to '269'
startdir = 230
lastdir = 270
dirnum = lastdir - startdir
for i in range(0,dirnum//2):

    t_id = str(i + startdir).zfill(3)
    r_id = str(i + startdir + (dirnum)//2).zfill(3)
    save_id = str(i).zfill(3)

    t_root = "REDS_dataset/train_sharp/" + t_id + "/"
    r_root = "REDS_dataset/train_sharp/" + r_id + "/"

    is_saving = True


    mask_w = 1920
    mask_h = 1080
    w = 1280
    h = 720

    g_mask=gkern(mask_w,3)
    g_mask = cv2.resize(g_mask, (mask_w, mask_h))
    g_mask=np.dstack((g_mask,g_mask,g_mask))

    k_sz=np.linspace(1,5,80) # for synthetic images
    sigma=k_sz[np.random.randint(0, len(k_sz))]

    t_images, r_images = prepare_data(t_root, r_root) # get image path

    if t_images == []:
        print("t_images don't exist.")
        sys.exit()
    elif r_images == []:
        print("r_images don't exist.")
        sys.exit()

    images_len = min(len(t_images), len(r_images))

    const_att=1.08+np.random.random()/10.0
    neww=np.random.randint(0, mask_w-w)
    newh=np.random.randint(0, mask_h-h)
    alpha2 = 1-np.random.random()/5.0
    alpha1=g_mask[newh:newh+h,neww:neww+w,:]


    print('T:{}, R:{}'.format(t_id, r_id))
    count = 0
    for id in tqdm(range(0, images_len)):

        t_image = cv2.cvtColor(cv2.imread(t_images[id]), cv2.COLOR_BGR2RGB)
        r_image = cv2.cvtColor(cv2.imread(r_images[id]), cv2.COLOR_BGR2RGB)
        
        if ((t_image is not None) and (r_image is not None)):

            t_image = t_image.astype('float64')/255.0
            r_image = r_image.astype('float64')/255.0
            
            out_t, r_blur_mask, r_blur, blended = syn_data(t_image, r_image, sigma, alpha1, alpha2)

            if is_saving == True:
                os.makedirs('test_RR/' + save_id, exist_ok=True)
                plt.imsave("test_RR/" + save_id + "/{:0=8}.png".format(count), blended)
                os.makedirs('test_GT/' + save_id, exist_ok=True)
                plt.imsave("test_GT/" + save_id + "/{:0=8}.png".format(count), out_t)
                os.makedirs('test_R_blur_mask/' + save_id, exist_ok=True)
                plt.imsave('test_R_blur_mask/' + save_id + "/{:0=8}.png".format(count), r_blur_mask)

        else:
            print("t_image or r_image is None.")
        count += 1


time_end = time.perf_counter()
tim = time_end- time_sta
print('{}s'.format(tim))

