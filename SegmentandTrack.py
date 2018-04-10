
import string
import sys
import os
import time
import pdb
#######################################################################
#######################################################################

# helper function for input
def text_input(DISPLAY):
	str = ''
	while (str == ''):
		str = raw_input(DISPLAY)	
	return str
	
# helper function for input
def bool_input(DISPLAY):
	bValue = None
	while (bValue is None):
		str = raw_input(DISPLAY)
		if (str=='y' or str=='Y'):
			bValue = True
		elif (str=='n' or str=='N'):
			bValue = False
	return bValue


# helper function for input
def int_input(DISPLAY):
	iValue = None
	while (iValue is None):
		str = raw_input(DISPLAY)
		try:
			iValue = int(str)
		except:
			iValue = None
	return iValue
	

# helper function for input
def float_input(DISPLAY):
	iValue = None
	while (iValue is None):
		str = raw_input(DISPLAY)
		try:
			iValue = float(str)
		except:
			iValue = None
	return iValue
####################################################################
####################################################################

def getfilename(prompt):
	ISFILE = False
	while not ISFILE:
		FILENAME = text_input(prompt)
		ISFILE = os.path.isfile(FILENAME)
		if not ISFILE:
			print 'cannot find file'
	return FILENAME
	
def getdirname(prompt):
	ISPATH = False
	while not ISPATH:
		PATHNAME = text_input(prompt)
		ISPATH = os.path.isdir(PATHNAME)
		if not ISPATH:
			print 'cannot find directory'
	return PATHNAME

####################################################################
####################################################################
#import custom python scripts
import Image_alignment
import Image_analysis
import RunWeka
import TrackCellLineages


b_ALIGN = bool_input('Do you wish to Align images (Y/N): ')
b_Segment = bool_input('Do you wish to train and/or apply a classifier (Y/N): ')
b_track = bool_input('Do you wish to track cells (Y/N): ')
b_ANALYZE = bool_input('Do you wish to analyze whole image fluorescence (Y/N): ')
if b_ANALYZE:
	b_RENDER = bool_input('Do you wish to render videos (Y/N): ')
else:
	b_RENDER = False

WorkDir = os.getcwd()
print 'current working directory is ', WorkDir
ImageDir = getdirname('What is the name of the image directory, relative to the working directory (e.g. Practice): ')

if b_Segment or b_track:
	FIRSTFRAME = int_input('First frame in dataset (e.g. 448): ')
	FRAMEMAX = int_input('Last frame in dataset (e.g. 467): ')
	
if b_track or b_ANALYZE:
	Ftime = float_input('What is the time per frame in minutes (e.g. 0.5): ')
	
if b_ALIGN or b_track or b_ANALYZE:
	FLINITIAL = int_input('First frame with fluorescence image (e.g. 449): ')
	FLSKIP = int_input('Number of frames between fluorescence images (i.e. every nth image): ')	

FNAME = False
while not FNAME:
	fname = text_input('Name of image files preceding channel and filenumber (e.g. 20171212_book): ')
	ftest = ImageDir + '/' + fname + '-%s-%03d.tif'
	FNAME = os.path.isfile(ftest %('p', FIRSTFRAME))
	if not FNAME:
		print 'cannot find file: ', ftest %('p', FIRSTFRAME)
		continue
	if b_ALIGN or b_track or b_ANALYZE:
		FNAME = os.path.isfile(ftest %('g', FLINITIAL))
		if not FNAME:
			print 'cannot find file: ', ftest %('g', FLINITIAL)

if b_ANALYZE or b_track:
	FLChannels = 1
	FLLABELS = []
	for i in range(FLChannels):
		label = text_input('Enter name of fluorescense channel ' + str(i+1) +' (e.g. GFP): ')
		FLLABELS.append(label)

if b_ALIGN:
	ROIALIGNFILE = bool_input('Do you have a csv file for a stationary area (Y/N): ')
	if ROIALIGNFILE:
		AlignROI = getfilename('Enter the path (relative to the working directory) to the csv file (e.g. Align_roi.csv): ')
	else:
		AlignROI = None
	AlignDir = text_input('Name of directory for output images (e.g. Aligned): ')
else:
	AlignDir = ImageDir
		
if not b_Segment:
	GETMASKS = bool_input('Do you have masks for the images (Y/N): ')
	if GETMASKS:
		MASKDIR = False
		while not MASKDIR:
			Mask2Dir = text_input('Enter directory containing masks relative to ' + AlignDir + ': ')
			MASKDIR = os.path.isdir(AlignDir + '/' + Mask2Dir)
			if not MASKDIR:
				print 'could not find directory'
	else:
		Mask2Dir = None			

if b_ALIGN:
	AlignARG = [AlignROI, AlignDir, ImageDir, fname, FLINITIAL, FLSKIP]
	Image_alignment.run(AlignARG)


if b_Segment:
	IJPATH = False
	while not IJPATH:
		IMAGEJ = text_input('Enter absolute path to ImageJ executable file (e.g. /home/user/Downloads/Fiji.app/ImageJ-linux64): ')
		IJPATH = os.path.isfile(IMAGEJ)
		IJEX = os.access(IMAGEJ,os.X_OK)
		if not IJPATH and IJEX:
			print 'could not find application'
	def training():
		#argument to use with RunWeka.training
		WekaARG1 = [IMAGEJ, WorkDir + '/']
		openFiji = False
		while not openFiji:
			print 'Please open an instance of Fiji. '
			time.sleep(2)
			openFiji = bool_input('Is there an open instance of Fiji (Y/N): ')
			if not openFiji:
				print 'Please open Fiji'
		RunWeka.training(WekaARG1)
	
	ROUND2 = int_input('How many round of classification are you running (1 or 2): ')
	if 'linux' in IMAGEJ:
		PROCESS = bool_input('Would you like to batch classify the images in the background (it is faster; Y/N): ')
	else:
		PROCESS = False
	if PROCESS:
		CORES = int_input('How many processes are available to use for multiprocessing (set to 1 for no multiprocessing): ')
	else:
		CORES = None
	if ROUND2 == 2:
		
		TRAINING = bool_input('Do you have a trained classifier (Y/N): ')
		if not TRAINING:
			#train first classifier
			training()
		
		#first batch classification
		classifierfile1 = getfilename('Enter path to classifier relative to working directory (e.g. Aligned/classifier.model): ')
		Mask1Dir = text_input('Enter name of directory to save masks within ' + AlignDir + ' (e.g. Mask1): ')
		runMask1Dir = AlignDir + '/' + Mask1Dir
		
		if not os.path.isdir(runMask1Dir):
			os.system('mkdir ' + runMask1Dir)
		else:
			os.system('rm ' + runMask1Dir + '/' + fname + '*.png')
		WekaARG2 = [IMAGEJ, True, WorkDir + '/', AlignDir+ '/', runMask1Dir + '/', FIRSTFRAME, FRAMEMAX, fname + '-p', 'tif', classifierfile1, CORES] 
		RunWeka.batchsegment(WekaARG2)
		
		print 'continuing to second round of classification\n'
	else:
		runMask1Dir = getdirname('Enter directory containing images to classify relative to the working directory (e.g. Aligned/Mask1): ')
		
	TRAINING2 = bool_input('Do you have a second trained classifier (Y/N): ')
	if not TRAINING2:
		#train first classifier
		training()
	
	#second batch classification
	classifierfile2 = getfilename('Enter path to classifier relative to working directory (e.g. Aligned/classifier2.model): ')
	Mask2Dir = text_input('Enter name of directory to save masks within ' + AlignDir + ' (e.g. Mask2): ')
	runMask2Dir = AlignDir + '/' + Mask2Dir
	
	if not os.path.isdir(runMask2Dir):
		os.system('mkdir ' + runMask2Dir)
	else:
		os.system('rm ' + runMask2Dir + '/' + fname + '*.png')
	WekaARG2 = [IMAGEJ, False, WorkDir + '/', runMask1Dir+ '/', runMask2Dir + '/', FIRSTFRAME, FRAMEMAX, fname + '-p', 'png', classifierfile2, CORES] 
	RunWeka.batchsegment(WekaARG2)

	
if b_track:
	if not Mask2Dir == None:
		LineageDir = 'Lineages'
		if not os.path.isdir(AlignDir + '/' + LineageDir):
			os.system('mkdir ' + AlignDir + '/' + LineageDir)
		AREAMIN = int_input('Enter the minimum cell area for tracking (e.g. 100): ')
		AREAMAX = int_input('Enter the maximum cell area for tracking (e.g. 2500): ')
		MINTRAJLENGTH = int_input('Enter the minimum number of frames to track cells through (e.g. 15): ')
	else:
		b_track = False
		print ('tracking cells requires a mask directory')

if b_ANALYZE or b_RENDER:
	ROIANALYZE = bool_input('Do you wish to analyze a region of interest (Y/N): ')
	CroptoROI = bool_input('Do you want to crop the images based on an ROI (Y/N): ')
	if ROIANALYZE or CroptoROI:
		ROIFILE = getfilename('Enter the path (relative to the working directory) to the csv file for the ROI to analyze (e.g. ROI.csv): ')
	else:
		ROIFILE = None
	Writelineagetext = bool_input('Do you want to number the cells in the images based on lineage tracking (Y/N): ')
	if not Mask2Dir == None:
		ContourImage = bool_input('Do you want to contour cells based on masks (Y/N): ')
	else:
		ContourImage = False
		
if b_track:
	TrackARG = [AlignDir, fname, Mask2Dir, LineageDir, FIRSTFRAME, FRAMEMAX, AREAMIN, AREAMAX,FLLABELS, Ftime, FLSKIP, FLINITIAL,MINTRAJLENGTH]
	TrackCellLineages.run(TrackARG)
	
if b_ANALYZE or b_RENDER:
	if Writelineagetext:
		WRITETEXT = os.path.isfile('lineagetext.pkl')
		if not WRITETEXT:
			print 'cannot find lineagetext.pkl to use for writing text'
			Writelineagetext = False
	
	AnalyzeARG = [AlignDir, FLSKIP, b_ANALYZE, ROIFILE, b_RENDER, ContourImage, AlignDir + '/' + Mask2Dir, CroptoROI, Writelineagetext, FLChannels, FLINITIAL, fname, Ftime, FLLABELS]
	Image_analysis.run(AnalyzeARG)
