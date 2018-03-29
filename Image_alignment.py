#For aligning phase and fluorescence images. 
#Uses Fast Fourier Transform (FFT) to calculate alignment for a series of images, and outputs aligned images. 
#To improve alignment, a region of interest can be input for use during FFT calculations. 

import numpy as np

import glob

import sys,os

import cv2 as cv
from scipy.ndimage.filters import gaussian_filter


from StringIO import StringIO

import matplotlib as mpl
# mpl.use('Agg')  # optimization for faster rendering than Agg (at least, it is supposed to be)
mpl.rc('pdf', fonttype=42)   # to embed fonts in PDF
import matplotlib.pyplot as plt
plt.ion()
plt.show()

from numpy.fft import fft2,ifft2

import pdb

#######################################################################
#######################################################################
#change the following based on the working directory

#a csv file containing a region without cells throughout the images
ROIFILE = 'Align_roi.csv'

# output image directory
OUTDIR = 'Aligned'

# input image directory
INDIR = 'Practice'

#all the phase images in the INDIR that contain the given string
FILES = glob.glob(INDIR + '/*p*')

#the name of the fluorescence files
#currently the script requires the difference between phase images contain -p- and fluorescence contain -g-
flname = INDIR + '/20171212_book-g-%03d.tif'

#change the following based on your imageset
#first frame with fluorescence data
FLINITIAL = 449
#frames between fluorescecne images (every nth frame)
FLSKIP = 10
 
CROPIMAGE = True

#end of parameters requiring modification
#######################################################################
#######################################################################
  
# the index (in the FILES list) to use for a reference
REFIND = 0


#gets the number of files
NFILES = len(FILES)
#the following is so that fluorescence images are shifted as well
FLFILES = []
#loop parameters should be for every nth image that is shifted
for flfile in range (FLINITIAL,NFILES+FLINITIAL,FLSKIP):
    flnamed = flname % flfile
    FLFILES.append(flnamed)
#######################################################################
#######################################################################

# image loading functions

# load an image, rescale it, and return it
def imgLoad(fname,scale):
	img = cv.imread(fname,cv.IMREAD_ANYDEPTH)
	#rarely, imread fails, so we try again
	if (img is None):
		img = cv.imread(fname,cv.IMREAD_GRAYSCALE)
	img = scaleDown(img,scale)
	return img

# scale an image and return it
def scaleDown(img,scale):
	newx,newy = int(img.shape[1]*scale),int(img.shape[0]*scale)
	return cv.resize(img, (newx,newy))


####################################################################
####################################################################
#make array from ROIFile

#print 'TRYING TO IMPORT ROI:', ROIFILE
ROITXT = open(ROIFILE, 'rb').read()

# trick for newline problems
ROITXT = StringIO(ROITXT.replace('\r','\n\r'))

ROI = np.loadtxt(ROITXT, delimiter=",",skiprows=1, dtype='int')
#print 'IMPORTED ROI\'S'

ROI = np.reshape(ROI, (-1,5))

#######################################################################
#######################################################################


#take ROI
def getROI(img,ROI,bDisplay=False):
    
    DIMS = img.shape
    
    ROI_xy = ROI[:,1:]
    #print ROI_xy
    
    # ROI for full image
    ROI_full = np.array([0,0,DIMS[0],DIMS[1]], dtype=np.int)

    ROI_xy = np.array(ROI_xy, dtype=np.int)
    ROI_xy = np.insert(ROI_xy, 0, ROI_full, axis=0)
    
    for roi in ROI_xy:
        # a subimage for a specific ROI
        imgroi = img[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]

    return imgroi


#######################################################################
#######################################################################

# perform (effectively) PIV for the final correction
def FFTCompare(img1,img2,bDisplay=False):
    
    if not ROIFILE == None:
        #use ROIs
        img1 = getROI (img1, ROI)
        img2 = getROI (img2, ROI)
    
    # keep original images
    img1_0 = img1*1
    img2_0 = img2*1
    
    # edge filter
    #img1 = absGradient(img1,1.0)
    #img2 = absGradient(img2,1.0)
    
    # subtract mean
    img1 = img1-np.mean(img1)
    img2 = img2-np.mean(img2)
    
    # fast convolution
    fconv = ifft2(np.conj(fft2(img1))*fft2(img2))
    fconv = np.real(fconv)
    
    # force zero convolution beyond a certain distance
    # make this parameter big to allow for large shifts
    RADIUS = 500
    SZ = fconv.shape
    fconv[RADIUS:SZ[0]-RADIUS,:] = 0.0
    fconv[:,RADIUS:SZ[1]-RADIUS] = 0.0
    
    
    # find translation vector
    diff = np.array(np.unravel_index(fconv.argmax(),fconv.shape))
    
    
    # return images back to non-edge form (for visualization)
    img1 = img1_0
    img2 = img2_0
    

    # if true, plot feedback
    if (bDisplay):
        # correct image for visualization
        img3 = np.roll(img2,-diff,axis=(0,1))

        plt.figure(figsize=(12,12))

        plt.subplot('221')
        plt.imshow(img1, cmap='gray')
        plt.title('image 1')

        plt.subplot('222')
        plt.imshow(img2, cmap='gray')
        plt.title('image 2')

        plt.subplot('223')
        plt.imshow(img2, cmap='gray', clim=[np.percentile(img2,1),np.percentile(img2,99)])
        plt.imshow(np.ma.masked_where(img1<np.percentile(img1,80),img1), cmap='hot',alpha=0.5,                                      clim=[np.percentile(img1,5),np.percentile(img1,95)+2*(np.percentile(img1,95)-np.percentile(img1,5))])
        plt.title('before correction')

        plt.subplot('224')
        plt.imshow(img3, cmap='gray', clim=[np.percentile(img3,1),np.percentile(img3,99)])
        plt.imshow(np.ma.masked_where(img1<np.percentile(img1,80),img1), cmap='hot',alpha=0.5,                                      clim=[np.percentile(img1,5),np.percentile(img1,95)+2*(np.percentile(img1,95)-np.percentile(img1,5))])
        plt.title('after correction')

        plt.tight_layout()
        
    # shift difference vector to be negative when appropriate
    SZ = img1.shape
    if (diff[0]>SZ[0]/2.0):
        diff[0]-=SZ[0]
    if (diff[1]>SZ[1]/2.0):
        diff[1]-=SZ[1]
    
    # return answer
    return diff

#######################################################################
#######################################################################


def fixImage(img,diff):

    diff = diff * -1
    img = np.roll(img,diff,axis=(0,1))

    if (diff[0]>0):
        img[0:diff[0],:]=0

    if (diff[0]<0):
        img[diff[0]:,:]=0
		
    if (diff[1]>0):
        img[:,0:diff[1]]=0

    if (diff[1]<0):
        img[:,diff[1]:]=0



    return img

#######################################################################
#######################################################################

# compute difference vectors for all files
#keep largest and smallest changes to use for cropping images
Bdiff = [0,0]
Sdiff = [0,0]

print ('calculating image shift')
DIFF = dict()
for fname in FILES:
    diff = FFTCompare(imgLoad(FILES[REFIND],1.0), imgLoad(fname,1.0))
    #print(diff)
    DIFF[fname] = diff
    if diff[0] > Bdiff[0]:
	Bdiff[0] = diff[0]
    if diff[0] < Sdiff[0]:
	Sdiff[0] = diff[0]
    if diff[1] > Bdiff[1]:
	Bdiff[1] = diff[1]
    if diff[1] < Sdiff[1]:
	Sdiff[1] = diff[1]

#pdb.set_trace()
print ('done')

#######################################################################
#######################################################################
# make directory if it does not exist
if (not os.path.isdir(OUTDIR)):
    os.system('mkdir ' + OUTDIR)

print ('shifting images')

if CROPIMAGE:
	#make even width and length
	if Bdiff[0]%2 == 1:
		Bdiff[0] = Bdiff[0] +1
	if Bdiff[1]%2 == 1:
		Bdiff[1] = Bdiff[1] +1
	if Sdiff[0]%2 == 1:
		Sdiff[0] = Sdiff[0] - 1
	if Sdiff[1]%2 == 1:
		Sdiff[1] = Sdiff[1] - 1
	Bdiff[0] = Bdiff[0] * -1
	Bdiff[1] = Bdiff[1] * -1
	Sdiff[0] = Sdiff[0] * -1
	Sdiff[1] = Sdiff[1] * -1

for fname in FILES:
    img = imgLoad(fname,1.0)
    df = DIFF[fname]
    fname_stripped = fname.split('/')[-1]
    #pdb.set_trace()
    img = fixImage(img,df)
    if CROPIMAGE:
		ishape = img.shape
		img = img[Sdiff[0]:(ishape[0]+Bdiff[0]),Sdiff[1]:(ishape[1]+Bdiff[1])]
    cv.imwrite(OUTDIR + '/' + fname_stripped, img)
    #print df
    gname = fname.replace ('-p-','-g-')
    if gname in FLFILES:
        gimg = imgLoad(gname,1.0)
        gimg = fixImage(gimg,df)
        if CROPIMAGE:
			gishape = img.shape
			gimg = gimg[Sdiff[0]:(ishape[0]+Bdiff[0]),Sdiff[1]:(ishape[1]+Bdiff[1])]
        gname_stripped = gname.split('/')[-1]
        cv.imwrite(OUTDIR + '/' + gname_stripped, gimg)
	#pdb.set_trace()
print ('done')




		
		
		
		
		
