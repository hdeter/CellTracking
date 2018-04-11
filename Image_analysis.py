
#for use with image datasets to measure fluorescence for the global image or a region of interest
#also renders videos from phase and fluorescence images
#can use output from track-cell-lineages.py to number cells by lineage

import string
import sys

import cv2 as cv
import numpy as np
from PIL import Image
import pdb
from subprocess import call
from os import remove, path
import pickle

from StringIO import StringIO
from matplotlib import pyplot as plt
import matplotlib as mpl

from scipy import convolve

import os

import scipy.ndimage as ndimage
import scipy.misc

#########################################################################
########################################################################
#########################################################################
########################################################################
def main(argv):
	#Change the following based on your directories and dataset (follow the comments)


	#Set to the name of the experiment directory as a string (ex: 'practice')
	#directory should be in the working directory (the directory the script is in)
	EXPT_NAME = argv[0]

	#names for saving the data later
	EXPT_NAME_DATA = 'data_' + EXPT_NAME
	EXPT_NAME_PKL = 'data_' + EXPT_NAME + '.pkl'

	#number of frames between fluorescence data
	frameSkip = argv[1]

	#if you want to analyze the files (measure fluorescence) set to True; if not set to False
	b_ANALYZE = argv[2]
	print

	#Path to csv file (relatvie to working directory) with the ROI file for analysis as a string (ex: 'ROI_file'); 
	#if there is no ROI file set to None
	#ROI file is XY#,BX,BY,W,H in pixels
	ROIFILE = argv[3]


	#If you want to render images set to True; if not set to False
	b_RENDER = argv[4]
	print

	#If you have masks and want to contour the image set ContourImage to True
	ContourImage = argv[5]
	#Path to the directory containing the masks relative to the working directory
	MaskDirectory = argv[6]

	#crops the image based on csv file if set to True
	CroptoROI = argv[7]
	#path to csv file containing ROI relative to working directory
	ROICROPFILE = argv[3]
	if ROICROPFILE == None:
		CroptoROI = False

	#If you wish to label cells based on lineage tracking set to True
	Writelineagetext = argv[8]
	#picklefile with lineage text (output by Track-cell-lineages.py)
	Lineagetextfile = 'lineagetext.pkl'

	# Set to number of channels, including phase (phase+gfp= 2)
	CMAX = argv[9] + 1

	#first frame that has fluorescence data
	FRAMEMIN = argv[10]

	# parameters for images
	#
	MaxC = CMAX                               # max number of fluorescence channels in image data
	AIfmtInFile = argv[11] + '-%s-%03d.tif'       # format string for data
	imgScale = 1                                 # scale to import images

	#framerate for final video (only includes fluorescence frames). Start with 10
	FRAMERATE = 10


	#parameters for analyzing multiple xy regions; currently set to 1

	#If rendering the files, set to XY region to be used as a reference; if not, set to 1
	XYREF = 1
		
	# Set to the number of XY regions 
	XYMAX = 1
	print

	# Set to a list of XY regions to be rendered 
	#ex: XYRENDER = [1,2,3,4,5,6,7,8,9]
	XYRENDER = [1]
	XYMax = 1                                      # max value of xy position

	#########################################################################
	########################################################################
	#########################################################################
	########################################################################

	#sets up commands to run at the end of the script
	#this section contains parameters for video rendering
	#Go through comments and adjust as desired to change video rendering
	# only use these for setting brightness, etc/
	#note that 1 is the phase image and 2-4 are fluorescence channels

	if (b_RENDER):
		
		CONFIGFILE = 'config_P_Y_C_M'

		########################################
		########################################

		# set to True to render microscopy videos
		bRenderMicroscopyVideo = True

		# set to True to render plot videos
		bRenderROIPlotVideo = True


		########################################
		########################################

		#
		# parameters for images
		#
		CONFIGVARSimgScale = 1                 # scale to import images

		########################################
		########################################

		# the real time per frame, in minutes
		# often 0.5 minutes, or 30 seconds
		timePerFrame = argv[12];

		########################################
		########################################


		# used to color different channels
		# if all zero, then this equates to not coloring the image with the channel
		#(blue, green, red) use range of 0 to 1
		CONFIGVARSimg_rhoC = dict()
		CONFIGVARSimg_rhoC[1] = (0.33,0.33,0.33)
		CONFIGVARSimg_rhoC[2] = (0.0,1.0,0.0)
		CONFIGVARSimg_rhoC[3] = (1.0,0.0,0.0)
		CONFIGVARSimg_rhoC[4] = (0.0,0.0,1.0)


		# used to scale different channels when coloring
		# all values are scaled by the value variation of the channel (across all time)
		## used to shift the image values (zero means no shift; + is darker; - is brighter)
		CONFIGVARSimg_shift = dict()
		CONFIGVARSimg_shift[1] = -1.0
		CONFIGVARSimg_shift[2] = 1
		CONFIGVARSimg_shift[3] = 1
		CONFIGVARSimg_shift[4] = 0.0

		## used to scale the image values (1.0 means no scaling; 0 excludes given channel)
		CONFIGVARSimg_scale = dict()
		CONFIGVARSimg_scale[1] = 1.0
		CONFIGVARSimg_scale[2] = 0.015
		CONFIGVARSimg_scale[3] = 1.0
		CONFIGVARSimg_scale[4] = 1.0

		########################################
		########################################
		
		FCLABELS = argv[13]
		# used to give labels to various channels (used in plotting)
		
		CONFIGVARSimg_label = dict()
		CONFIGVARSimg_label[1] = 'phase contrast'
		for i in range(len(FCLABELS)):
			CONFIGVARSimg_label[i+2] = FCLABELS[i]

		########################################
		########################################

		# the following code sets the filter for the plot of each channel.
		# Index of plot_filterIndex is for the channel.
		#
		# filter 0: boring filter (no filter)
		# filter 1: filter to set minimum to zero and standard deviation to 1
		# filter 2: high pass filter, subtracts running average of 10 frames, normalizes standard deviation to 1
		# filter 3: filter to set the minimum to zero and maximum to 1
		#
		plot_filterIndex = dict()
		plot_filterIndex[1] = 0
		plot_filterIndex[2] = 1
		plot_filterIndex[3] = 1
		plot_filterIndex[4] = 1

		########################################
		########################################

		# y-axis limits for various filters.
		# Index for YLIM is for the filter index.

		# YOU MAY WANT TO CHANGE YLIM[1] OR YLIM[2], BUT THE OTHERS ARE LIKELY OK

		# "None" means autoscaling
		CONFIGVARSYLIM = dict()
		CONFIGVARSYLIM[0] = None
		CONFIGVARSYLIM[1] = [-0.2,5.0]
		CONFIGVARSYLIM[2] = [-5.0,5.0]
		CONFIGVARSYLIM[3] = [-0.2,1.2]

		########################################
		########################################

		# sets the filtering window size for filter2 (10 is a good standard number)
		CONFIGVARSFILTERWINDOW = 10

		########################################
		########################################

		# plot window size for time
		CONFIGVARSPLOTWINDOWSIZE = [-20.0,20.0]

		# label for x-axis
		CONFIGVARSTIMELABEL = 'time (min)'

		########################################
		########################################

		# format string for output files
		if (not os.path.isdir('out')):
			os.system('mkdir ' + 'out')
		CONFIGVARSfmtOutDIR = './out/'
		CONFIGVARSfmtOutFile = 'frame%06d.png'
		CONFIGVARSfmtOutFileAll = CONFIGVARSfmtOutDIR + CONFIGVARSfmtOutFile

		########################################
		########################################
	########################################

	#########################################################################
	########################################################################
	#########################################################################
	########################################################################

	# filters plots to normalize them

	# filter 0: boring filter (no filter)
	def filterData0(data):
		return data*1.0

	# filter 1: filter to set minimum to zero and standard deviation to 1
	def filterData1(data):
		data = data - np.min(data)
		data = data / np.std(data)
		return data*1.0

	# filter 3: filter to set the minimum to zero and maximum to 1
	def filterData3(data):
		data = data - np.min(data)
		data = data / np.max(data)
		return data*1.0

	# filter 2: high pass filter, subtracts running average of 10 frames, normalizes standard deviation to 1
	# in the future - make these easy to pass as parameters
	def filterData2(data):
		#pdb.set_trace()
		WIND = CONFIGVARSFILTERWINDOW
		data = data-convolve(data, np.ones((WIND))/WIND, mode='same')
		#data = data-np.min(data[50:100])

		data[0:WIND] = 0.0
		data[-WIND:] = 0.0

		#data = data / (np.std(data[50:300]))
		data = data / (np.std(data))

		return data
		
	#########################################################################
	########################################################################
	#########################################################################
	########################################################################

	#from imageToolsMather

	# load an image, rescale it, and return it
	def imgLoad(fname,scale):
		img = cv.imread(fname,-1)
		img = scaleDown(img,scale)
		return img

	# scale an image and return it
	def scaleDown(img,scale):
		newx,newy = int(img.shape[1]*scale),int(img.shape[0]*scale)
		return cv.resize(img, (newx,newy))

	# change the brightness of an image
	def shiftBrightness(img,minVal,scaleVal):
		img = (img-minVal)*scaleVal
		#img = np.maximum(img,0)
		return img

	# add grayscale to color image, with a weight
	def addGrayToColor(IMG,img,rho):
		for ii in range(3):
			IMG[:,:,ii] = IMG[:,:,ii] + img*rho[ii]
		return IMG


	#########################################################################
	########################################################################
	#########################################################################
	########################################################################

	# From imageToolsMather
	# unpacks results into nice numpy arrays
	def unpackResults(RES):

		# list of frame numbers
		img_frame = []

		# several dictionaries for lists of image statistics
		img_mean = dict()
		img_std = dict()
		img_median = dict()
		img_min = dict()
		img_max = dict()

		# more dictionaries for higher order statistics
		img_global_mean = dict()
		img_global_std = dict()
		img_global_median = dict()
		img_global_min = dict()
		img_global_max = dict()



		# a dictionary for a list of file names
		img_fname = dict()


		index = 0	# frame index
		for res in RES:
			img_frame.append(res['frame'])
			stats = res['stats']


			indexc = 1	# channel index
			for statobj in stats:
				# just get stats
				stat = statobj['stats']


				# on first iteration, make a new list for each C channel
				if (index==0):
					img_mean[indexc] = []
					img_std[indexc] = []
					img_median[indexc] = []
					img_min[indexc] = []
					img_max[indexc] = []

					img_fname[indexc] = []


				# take particular statistics
				img_mean[indexc].append(stat['mean'])
				img_std[indexc].append(stat['std'])
				img_median[indexc].append(stat['median'])
				img_min[indexc].append(stat['min'])
				img_max[indexc].append(stat['max'])

				img_fname[indexc].append(statobj['fname'])


				# increment channel index
				indexc = indexc+1

			# increment frame index
			index = index + 1



		# turn results into np arrays (for certain data)
		for key in img_mean.keys():
			img_mean[key] = np.array(img_mean[key])
			img_std[key] = np.array(img_std[key])
			img_median[key] = np.array(img_median[key])
			img_min[key] = np.array(img_min[key])
			img_max[key] = np.array(img_max[key])

			img_global_mean[key] = np.mean(img_mean[key])
			img_global_std[key] = np.mean(img_std[key])
			img_global_median[key] = np.mean(img_median[key])
			img_global_min[key] = np.mean(img_min[key])
			img_global_max[key] = np.mean(img_max[key])


		# make a final dictionary to return that holds all the relevant info
		statNP = dict()

		statNP['frame'] = img_frame

		statNP['mean'] = img_mean
		statNP['std'] = img_std
		statNP['median'] = img_median
		statNP['min'] = img_min
		statNP['max'] = img_max

		statNP['mean_mean'] = img_global_mean
		statNP['mean_std'] = img_global_std
		statNP['mean_median'] = img_global_median
		statNP['mean_min'] = img_global_min
		statNP['mean_max'] = img_global_max

		statNP['fname'] = img_fname


		return statNP


	#########################################################################
	########################################################################
	#########################################################################
	########################################################################

	# Analyze images

	# list of stats
	stat_keys = ['min','max','std','mean','median']

	# associated functions for the stats
	stat_funcs = dict()
	stat_funcs['min'] = np.min
	stat_funcs['max'] = np.max
	stat_funcs['std'] = np.std
	stat_funcs['mean'] = np.mean
	stat_funcs['median'] = np.median


	# takes many different measurements for a particular DIR + XY region
	# input:
	#       DIR: directory of images
	#       XYLoc: xy number for image data
	#       ROI_xy: set of (rectangular) ROI's, within which statistics are taken
	def measureImagesXY(DIR, XYLoc, ROI_xy):

		#print 'experiment ' + DIR + ', Location ' + str(XYLoc)

		AIfmtInFileAll = DIR + '/' + AIfmtInFile    # format string for data

		###########################
		###########################

		# index variables to use in the main loop (below)
		#index must start at first frame with fluorescence data!
		index = FRAMEMIN
		outFrame = 0


		# master list for XY data
		imgStats_C_XY = []


		# loop until data is exhausted (for a given xy position)
		while True:

			#print 'xy = ', XYLoc, ',\t frame = ', index

			# array of file names (store for later use)
			fnames = []

			# array of image data (usually discard after each frame number)
			imgs = []

			# try to load image files
			# if there is an error, this indicates the data has reached an end
			try:
				for imgC in range(1,MaxC+1):
					if imgC == 1:
						imgC = 'p'
					elif imgC == 2:
						imgC = 'g'
					else:
						imgC = 'r'
					#print imgC
					fname1 = AIfmtInFileAll % (imgC, index)
					#print fname1                
					fnames.append(fname1)

					# load images
					# transpose image, because it is swapped compared to convection
					imgs.append(imgLoad(fname1,imgScale).astype(np.double).T)
			except:
				break



			# apply a local median blur to get rid of shot noise
			# get rid of this if the image is too grainy
			for imgC in range(0,MaxC):
				img = imgs[imgC]
				img = (cv.medianBlur(img.astype(np.uint16), 3)).astype(np.double)
				imgs[imgC] = img


			# make a variable for the size of the image
			DIMS = img.shape


			# ROI for full image
			ROI_full = np.array([0,0,DIMS[0],DIMS[1]], dtype=np.int)

			# if ROI is None, then make an ROI of the correct size, twice (for rendering purposes)
			if ROI_xy is None:
				ROI = np.ones((2,4), dtype=np.int)
				ROI[0,:] = ROI_full * 1
				ROI[1,:] = ROI_full * 1
			# otherwise, use existing ROI's scaled appropriately, plus add big ROI
			else:
				ROI = np.array(ROI_xy * imgScale, dtype=np.int)
				ROI = np.insert(ROI, 0, ROI_full, axis=0)

			#print ROI

			# take statistics across image channels
			# list of image stats for each channel
			imgStats_C = []
			for imgC in range(0,MaxC):
				img = imgs[imgC]
				# take statistics for a particular image with multiple ROI's

				# make an object for the statistics
				imgStats = dict()
				for key in stat_keys:
					imgStats[key] = []

				for roi in ROI:
					# a subimage for a specific ROI
					imgroi = img[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]

					#print '\tC = ', imgC, ',\t ROI = ', roi

					# actually take statistics
					for key in stat_keys:
						imgStats[key].append(stat_funcs[key](imgroi))

				# store image stats for this C channel
				# use a dict to store relevant details
				# save a dict for each C channel
				imgStatDict = dict()
				imgStatDict['fname'] = fnames[imgC]
				imgStatDict['ROI'] = ROI
				imgStatDict['scale'] = imgScale
				imgStatDict['stats'] = imgStats
				imgStats_C.append(imgStatDict)            


			# add stats to the master list of stats
			# use a dict to store data
			imgFrameDict = dict()
			imgFrameDict['frame'] = index
			imgFrameDict['stats'] = imgStats_C
			imgStats_C_XY.append(imgFrameDict)


			outFrame = outFrame+1
			# print outFrame


			# iterate to next frame
			index = index + frameSkip

			#break

		###########################
		###########################

		return (imgStats_C_XY)

		####################################################################
		####################################################################

	# the main function

	def analyzeImagesROI(argv):
		#global AIfmtInFile, XYMax, MaxC
		print 'parsing: ', argv	

		###################
		###################

		# only accept a single directory
		DIR = argv[0]

		# use next argument for number of XY positions
		XYMax = int(argv[1])

		# max number of C channels in image data
		MaxC = int(argv[2])

		print 'DIR: ' + DIR
		print 'XYMax: ' + str(XYMax)
		print 'MaxC: ' + str(MaxC)
		print 'FORMAT STRING: ' + AIfmtInFile

		###################
		###################

		if (len(argv)>=4):

			# load file containing ROI's to check
			try:
				ROIFILE = argv[3]
				print 'TRYING TO IMPORT ROI:', ROIFILE
				ROITXT = open(ROIFILE, 'rb').read()

				# trick for newline problems
				ROITXT = StringIO(ROITXT.replace('\r','\n\r'))

				ROI = np.loadtxt(ROITXT, delimiter=",",skiprows=1, dtype='int')
				print 'IMPORTED ROI\'S'
			except:
				ROI = None
		else:
			ROI = None

		#pdb.set_trace()

		###################
		###################


		# run through a number of XY regions
		OUTALL = dict()
		for XYLoc in range(1,XYMax+1):

			# if no ROI's, set the ROI's to None
			# otherwise, simply give the ROI's
			if (ROI is None or ROI.size == 0):
				ROI_xy = None
			else:
				#print ROI.shape
				ROI = np.reshape(ROI, (-1,5))
				#print ROI
				# determine if there are any ROI's for the current XY position
				ROI_xy_inds = np.where(ROI[:,0]==XYLoc)[0]

				# select only entries with the correct xy position
				ROI_xy = ROI[ROI_xy_inds,:]
				# get rid of xy data
				ROI_xy = ROI_xy[:,1:]
				print ROI_xy

			# get measurement results for a particular directory and XY position
			RES = measureImagesXY(DIR, XYLoc, ROI_xy)

			# append results to a master list
			OUTALL[XYLoc] = RES

			# periodically save statistics
			with open('data_' + DIR + '.pkl', 'wb') as fileoutput:
				pickle.dump(OUTALL, fileoutput, pickle.HIGHEST_PROTOCOL)

		#pdb.set_trace()


	#########################################################################
	########################################################################
	#########################################################################
	########################################################################


	#creates a contoured image based on mask
	def contouredimg(fimg, mask):
		img = mask
		img = 255-img
		#img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
		im2, contours, hierarchy = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
		contourimg = cv.drawContours(fimg*255, contours,-1, (0,0,65535),1)
		return contourimg


	#########################################################################
	########################################################################
	#########################################################################
	########################################################################

	#function to crop image based on ROI file

	def ROIcrop(img, ROICROPFILE, XYLoc):
		#import ROI file
		ROITXT = open(ROICROPFILE, 'rb').read()

		# trick for newline problems
		ROITXT = StringIO(ROITXT.replace('\r','\n\r'))

		ROI = np.loadtxt(ROITXT, delimiter=",",skiprows=1, dtype='int')
		#print 'IMPORTED ROI\'S'
		################################
		# make a variable for the size of the image
	#     DIMS = img.shape
		
	#     ROI_full = np.array([0,0,DIMS[0],DIMS[1]], dtype=np.int)
		
		ROI = np.reshape(ROI, (-1,5))
		# determine if there are any ROI's for the current XY position
		ROI_xy_inds = np.where(ROI[:,0]==XYLoc)[0]

		# select only entries with the correct xy position
		ROI_xy = ROI[ROI_xy_inds,:]
		# get rid of xy data
		ROI_xy = ROI_xy[:,1:]
		#print ROI_xy
		
	#     ROI = np.array(ROI, dtype=np.int)
	#     ROI = np.insert(ROI, 0, ROI_full, axis=0)
		
		imgroi = {}
		
		i=0
		
		for roi in ROI_xy:
			subroi = img[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]
			imgroi[i]= subroi
			i= i+1
				
		return imgroi

	#########################################################################
	########################################################################
	#########################################################################
	########################################################################

	#renderResults

	# renders a single image from various channel images
	def renderImage(statNP, statNP_ref, XY_loc, index, FRAMEMIN,TOWRITE):

		# change FRAMEMIN if not a positive number
		# FRAMEMIN being negative is a flag
		# could be buggy if used as a negative number in this code
		if (FRAMEMIN<0):
			FRAMEMIN = 0

		###############################
		###############################

		# get the file names from the results
		fnamesALL = statNP['fname']

		# frames associated with data
		frames = statNP['frame']

		# get the keys (channel numbers) from the results
		keys = fnamesALL.keys()

		# statistics on various channels
		img_mean = statNP_ref['mean_median']
		img_min = statNP_ref['mean_min']
		img_std = dict()
		for key in img_mean.keys():
			img_std[key] = statNP_ref['mean_median'][key] - statNP_ref['mean_min'][key]

		# names just for this one image
		fnames = dict()
		imgs = dict()

		# true once the composite color image is made
		bMadeColor = False

		for key in keys:
			# load the name
			fnames[key] = fnamesALL[key][index]

			# load the image data
			# note: image is transposed to fit normal coordinate system
			imgs[key] = imgLoad(fnames[key],1.0).astype(np.double)

			# make an image for the color reconstruction
			if (not bMadeColor):
				IMGDIM = imgs[key].shape
				IMG = np.zeros(IMGDIM + (3,));
				bMadeColor = True

			# perform a median blur for quality of rendering
			imgs[key] = (cv.medianBlur(imgs[key].astype(np.uint16), 3)).astype(np.double)

			# scale the image to a normal range
			imgs[key] = shiftBrightness(imgs[key]*1.0, img_min[key] + img_std[key]*CONFIGVARSimg_shift[key], CONFIGVARSimg_scale[key]/img_std[key])

			IMG = addGrayToColor(IMG, imgs[key], CONFIGVARSimg_rhoC[key])

		if ContourImage:
			#get name to use to load mask
			fname = fnamesALL[1][index]

			#import mask and contour image 
			fname = fname.replace(EXPT_NAME, MaskDirectory)
			fname = fname.replace('.tif','.png')
			#print fname
			mask = imgLoad(fname,1.0)
			IMG = contouredimg(IMG,mask)
		
		if Writelineagetext:
			try:
				for line in TOWRITE[FRAMEMIN+(index*frameSkip)]:
					XY = line[1]
					newtrajnum = line[0]
					cv.putText(IMG, newtrajnum, XY, cv.FONT_HERSHEY_DUPLEX, .3, (0,0,0), 1)
			except:
				print 'frame', index+1, 'has no lineage text'
		
		if CroptoROI:
			#fix this later to work with multiple ROI's
			cropimgs = ROIcrop(IMG, ROICROPFILE, XY_loc)
			IMG = cropimgs[0]
		
		#scale image
		IMG = scaleDown(IMG,CONFIGVARSimgScale)
		
		
		# add text
		cv.putText(IMG, ('XY%d, ' % XY_loc) + ('time = %0.01f min.' % (timePerFrame*(frames[index]-1))), (25,25), cv.FONT_HERSHEY_DUPLEX, 1.0, (1,1,1), 1)

		###############################
		###############################

		#print 'XY = ', XY_loc, ', frame = ', frames[index]

		###############################
		###############################

		
		# write image, finally
		# need to shift by lowest frame
		if ContourImage:
			cv.imwrite(CONFIGVARSfmtOutFileAll % (index-FRAMEMIN), IMG)
		else:
			cv.imwrite(CONFIGVARSfmtOutFileAll % (index-FRAMEMIN), IMG*255)

		#pdb.set_trace()

	####################################################################
	####################################################################




	# renders a single plot from various channel data (ctarget)
	def renderPlot(statNP1, statNP_all, XY_loc, index, ctarget, YLABEL, filterData, YLIM, pklfiletrim, FRAMEMIN):

		# change FRAMEMIN if not a positive number
		# FRAMEMIN being negative is a flag
		# could be buggy if used as a negative number in this code
		if (FRAMEMIN<0):
			FRAMEMIN = 0

		###############################
		###############################

		# statNP1 just for the current XY position

		# running array of data
		dataALLPlotted = []
		dataALLPlotted_orig = []

		plt.cla()
		xynow = 0
		for statNP in statNP_all:
			xynow = xynow + 1

			# get the file names from the results
			fnamesALL = statNP['fname']

			# frames associated with data
			frames = statNP['frame']

			# get the keys (channel numbers) from the results
			keys = fnamesALL.keys()


			# statistics on various channels
			img_mean = statNP['mean']


			data_x = (timePerFrame*(np.array(frames)-1))
			data_y = img_mean[ctarget]


			SZ = data_y.shape

			tcenter = timePerFrame*(frames[index]-1.0)
			plt.plot(np.array([0.0,0.0])+tcenter,[-10000,10000], 'r', linewidth=1)
			for targ in range(1,SZ[1]):

				dataALLPlotted_orig.append(1.0*data_y[:,targ])
				data = filterData(data_y[:,targ])

				if (not data is None):

					dataALLPlotted.append(data)
					plt.plot(data_x, data, '-', color='gray')

		dataALLPlotted = np.array(dataALLPlotted)
		#print 'shape = ', dataALLPlotted.shape

		#plt.plot(data_x, np.mean(dataALLPlotted, axis=0), 'b--', linewidth=2)
		plt.plot(data_x, np.median(dataALLPlotted, axis=0), 'r-', linewidth=2)
		#pdb.set_trace()
		
		##############################
		
		# extent of time rendering set in configuration file
		plt.xlim(np.array(CONFIGVARSPLOTWINDOWSIZE)+tcenter)

		if (not (YLIM is None)):
			plt.ylim(YLIM)

		plt.xlabel(CONFIGVARSTIMELABEL)
		plt.ylabel(YLABEL)
		plt.draw()

		plt.savefig(CONFIGVARSfmtOutFileAll % (index-FRAMEMIN))


		#print 'XY = ', XY_loc, ', frame = ', frames[index]


		###############################
		###############################

		# output to CSV

		# only output CSV on first rendering of plot
		if (index-FRAMEMIN==0):
			fnameCSV = pklfiletrim + '_xy' + str(XY_loc) + '_c' + str(ctarget) + '.csv'

			x = np.array(data_x)
			y1 = np.array(dataALLPlotted_orig)
			y2 = np.array(dataALLPlotted)

			y1m = np.median(y1, axis=0)
			y2m = np.median(y2, axis=0)

			SZy = y1.shape

			f = open(fnameCSV, 'wb')
			f.write('time (min),')
			f.write('unfiltered median FL,')
			f.write('filtered median FL,')
			for ijk in range(SZy[0]):
				f.write('unfiltered FL ROI %d,' % ijk)
			for ijk in range(SZy[0]):
				f.write('filtered FL ROI %d,' % ijk)
			f.write('\n')

			for ijk in range(SZy[1]):
				f.write(str(x[ijk]) + ',')
				f.write(str(y1m[ijk]) + ',')
				f.write(str(y2m[ijk]) + ',')

				for ijk2 in range(SZy[0]):
					f.write(str(y1[ijk2,ijk]) + ',')
				for ijk2 in range(SZy[0]):
					f.write(str(y2[ijk2,ijk]) + ',')
				f.write('\n')


			f.close()

		###############################
		###############################

		#pdb.set_trace()

	####################################################################
	####################################################################


	# the main function

	def renderResults(argv):

		# file containing results
		pklfile = argv[0]
		pklfiletrim = pklfile.replace('.pkl', '')

		# xy position to process
		XY_loc = int(argv[2])

		# xy position that sets color
		XY_loc_ref = int(argv[3])

		# abort early if current region is not in XY render
		if not XY_loc in XYRENDER:
			print 'NOT RENDERING XY%d' % XY_loc
			return


		# optional rendering parameters
		# pass as strings for compatibility
		FRAMEMIN = -1
		FRAMEMAX = -1
		if (len(argv)>5):
			# lowest frame number (NOT time) to render.  Default is 1.
			FRAMEMIN = int(argv[4])
			# highest frame number (NOT time) to render.  Default is the final frame.
			FRAMEMAX = int(argv[5])
		if not ((FRAMEMIN<0) or (FRAMEMAX<0)):
			print '!!! PARTIAL RENDERING !!!'

			# shift frames to "zero based" frames
			# first frame is actually zero inside this code
			# for the user, the first frame is 1
			FRAMEMIN -= 1
			FRAMEMAX -= 1


		###################
		###################

		# results for all xy positions
		with open(pklfile, 'rb') as fileinput:
			RESALL = pickle.load(fileinput)

		###################
		###################

		# results just for a single xy position
		RES = RESALL[XY_loc]
		# get results in a nicer numpy array format
		statNP = unpackResults(RES)


		# results just for a single xy position (for coloring)
		RES_ref = RESALL[XY_loc_ref]
		# get results in a nicer numpy array format
		statNP_ref = unpackResults(RES_ref)

		###################
		###################


		# results just for every xy position
		statNP_all = []
		if (XYRENDER == None):
			for key1 in RESALL:
				# get results in a nicer numpy array format
				statNP_all.append(unpackResults(RESALL[key1]))
		else:
			# only use partial results for plotting if specified
			for key1 in XYRENDER:
				# get results in a nicer numpy array format
				statNP_all.append(unpackResults(RESALL[key1]))

		###################
		###################

		# update some settings based on number of fluorescence channels

		# set the targets for plotting
		# do not plot phase constrast images (channel 1)
		#if CTARGETS is empty check image directories
		global CTARGETS
		
		CTARGETS = statNP_all[0]['mean'].keys()
		print 'CTARGETS', CTARGETS
		CTARGETS.remove(1)

		

		###################
		##################
		if Writelineagetext:
			with open(Lineagetextfile, 'rb') as f:
				TOWRITE = pickle.load(f)
		else:
			TOWRITE = None
			
		if (bRenderMicroscopyVideo == True):
			os.system('rm ' + CONFIGVARSfmtOutDIR + '*.png')

			# only render if parameters are correct
			for index in range(len(statNP['frame'])):
				if ((FRAMEMIN<0) or (FRAMEMAX<0)) or ((index>=FRAMEMIN) and (index<=FRAMEMAX)):
					renderImage(statNP, statNP_ref, XY_loc, index, FRAMEMIN,TOWRITE)

			ARG = 'avconv -y -framerate 10 -i ' + CONFIGVARSfmtOutFileAll + ' -c:v libx264 -pix_fmt yuv420p ' + pklfiletrim + ('_xy%d.mp4' % XY_loc)
			print 'running command: ' + ARG
			os.system(ARG)

		###################
		###################

		if (bRenderROIPlotVideo == False):
			for ctarget in CTARGETS:

				os.system('rm ' + CONFIGVARSfmtOutDIR + '*.png')

				#############
				#############

				# set filter and y-axis plotting limits, based on the configuration file
				filterIndex = plot_filterIndex[ctarget]
				YLIM = CONFIGVARSYLIM[filterIndex]

				if (filterIndex==0):
					filterData = filterData0
				elif (filterIndex==1):
					filterData = filterData1
				elif (filterIndex==2):
					filterData = filterData2
				elif (filterIndex==3):
					filterData = filterData3

				#############
				#############

				# set y-axis label for plots
				YLABEL = CONFIGVARSimg_label[ctarget]

				#############
				#############


				# only render if parameters are correct
				for index in range(len(statNP['frame'])):
					if ((FRAMEMIN<0) or (FRAMEMAX<0)) or ((index>=FRAMEMIN) and (index<=FRAMEMAX)):
						renderPlot(statNP, statNP_all, XY_loc, index, ctarget, YLABEL, filterData, YLIM, pklfiletrim, FRAMEMIN)


				ARG = 'avconv -y -framerate 10 -i ' + CONFIGVARSfmtOutFileAll + ' -c:v libx264 -pix_fmt yuv420p ' + pklfiletrim + ('_xy%d_%d.mp4' % (XY_loc,ctarget))
				print 'running command: ' + ARG
				os.system(ARG)


			os.system('mkdir CSV')
			os.system('mv data*.csv CSV/')


		###################
		###################

		#pdb.set_trace()

	#########################################################################
	########################################################################
	#########################################################################
	########################################################################

	#renderAll

	def renderAll(argv):
		print 'parsing: ', argv

		###################
		###################

		# import all arguments as strings

		# pickle file to use, e.g. 'data_QEntrain_212p_Run2.pkl'
		PICKLEFILE = argv[0]

		# configuration file, e.g. 'config_P_Y_C_M'
		CONFIG_MODULE = argv[1]

		# XY location to use for fluorescence scaling (same scaling for each video), e.g. '3'
		XYREF = argv[2]


		# optional rendering parameters
		# pass as strings for compatibility
		FRAMEMIN = '-1'
		FRAMEMAX = '-1'
		if (len(argv)>4):
			print '!!! PARTIAL RENDERING !!!'
			# lowest frame number (NOT time) to render.  Default is 1.
			FRAMEMIN = argv[3]
			# highest frame number (NOT time) to render.  Default is the final frame.
			FRAMEMAX = argv[4]


		###################
		###################

		# results for all xy positions
		with open(PICKLEFILE, 'rb') as fileinput:
			RESALL = pickle.load(fileinput)
		XYMAX = len(RESALL)

		#pdb.set_trace()

		###################
		###################

		for xyLOC in range(int(XYMAX)):
			XYLOC = str(1+xyLOC)
			renderResults([PICKLEFILE, CONFIG_MODULE, XYLOC, XYREF, FRAMEMIN, FRAMEMAX])


	#########################################################################
	########################################################################
	#########################################################################
	########################################################################


	ARG_ANALYZE = [EXPT_NAME, str(XYMAX), str(CMAX)]
	ARG_RENDER = [EXPT_NAME_PKL, CONFIGFILE, str(XYREF),str(FRAMEMIN)]


	if (not (ROIFILE is None)):
		ARG_ANALYZE.append(ROIFILE)

	print '... execution will run the following equivalent commands:'
	print

	if (b_ANALYZE):
		print string.join(ARG_ANALYZE, ' ')
	if (b_RENDER):
		print string.join(ARG_RENDER, ' ')


	#########################################################################
	########################################################################
	#########################################################################
	########################################################################

	#run commands based on answers in cell 2 as printed at the end of cell 3
	#fix this to run functions rather than python scripts



	if (b_ANALYZE):
		print
		print '====================================='
		print '========= analyzing images ========='
		print '====================================='
		print
		analyzeImagesROI(ARG_ANALYZE)

	if (b_RENDER):
		print
		print '===================================='
		print '========= rendering images ========='
		print '===================================='
		print
		renderAll(ARG_RENDER)

	print 'Finished'

####################################################################
####################################################################

# way to run as a module
def run(argv):
	main(argv)

####################################################################
####################################################################


if __name__ == "__main__":
	if len(sys.argv) > 1:
		main(sys.argv[1:])
	else:
		#True if analyzing whole image (optional: analyze ROI)
		b_ANALYZE = True
		#True if rendering videos (requires b_ANALYZE = True)
		b_RENDER = True

		#Directory containing the images
		ImageDir = 'Test'
		AlignDir = ImageDir
		#get working directory
		WorkDir = os.getcwd()
		#Image filename preceding channel indication (e.g. 20171212_book)
		fname = '20171212_book'

		#first frame of images
		FIRSTFRAME = 448
		#last frame of images
		FRAMEMAX = 467
		#first frame with fluorescence image
		FLINITIAL = 449
		#frequency of fluorescence images (i.e. every nth frame)
		FLSKIP = 10
		#time between frames in minutes
		Ftime = 0.5 #min
		#number of fluorescence channels
		FLChannels = 1

		#labels for fluorescence channels (must be strings)
		FLLABELS = ['GFP']
		
		#csv file containing ROI to analyze and/or crop to; if no file set to None
		ROIFILE = None
		#True if cropping images to ROI
		CroptoROI = False
		#True if writing lineages to video (requires lineagetracking.pkl)
		Writelineages = True
		#True if contouring images (requires Masks)
		ContourImage = False
		#mask directory (relative to image directory) for contouring images (only needed if ContourImage = True)
		Mask2Dir = 'Mask2'
		
	AnalyzeARG = [AlignDir, FLSKIP, b_ANALYZE, ROIFILE, b_RENDER, ContourImage, AlignDir + '/' + Mask2Dir, CroptoROI, Writelineages, FLChannels, FLINITIAL, fname, Ftime, FLLABELS]
	main(AnalyzeARG)

