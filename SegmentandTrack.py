
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
import Lineage_analysis


b_ALIGN = bool_input('Do you wish to align images? (Y/N): ')
b_Segment = bool_input('Do you wish to train and/or apply a classifier? (Y/N): ')
b_track = bool_input('Do you wish to track cells? (Y/N): ')

#only ask to output lineage csv if tracking pkl will be avaiable
if not b_track:
	b_tracked = os.path.isfile('lineagetracking.pkl')
if b_track or b_tracked:
	LANALYZE = bool_input('Do you wish to output csv files detailing data for individual lineages? (Y/N): ')
else:
	LANALYZE = False
	
b_ANALYZE = bool_input('Do you wish to analyze images (needed to get whole image fluorescence or render videos)? (Y/N): ')

#only ask to render videos if analysis file will be available
if b_ANALYZE:
	b_RENDER = bool_input('Do you wish to render videos? (Y/N): ')

WorkDir = os.getcwd()
print 'current working directory is ', WorkDir
ImageDir = getdirname('Enter the name of the image directory, relative to the working directory (e.g. Practice): ')


if not b_ANALYZE:
	if os.path.isfile('data_' + ImageDir + '.pkl'):
		b_RENDER = bool_input('Do you wish to render videos? (Y/N): ')
	else:
		b_RENDER = False
	

if b_ALIGN or b_Segment or b_track or b_ANALYZE or b_RENDER:
	FIRSTFRAME = int_input('Enter the number of the first frame in dataset (e.g. 448): ')



if b_Segment or b_track:
	FRAMEMAX = int_input('Enter the number of the last frame in dataset (e.g. 467): ')
	
if b_track or b_ANALYZE or b_RENDER:
	Ftime = float_input('Enter the time per frame in minutes (e.g. 0.5): ')
	
if b_ALIGN or b_track or b_ANALYZE or b_RENDER:
	FLINITIAL = int_input('Enter the number of the first frame with a fluorescence image (e.g. 449; for no fluorescence enter 0): ')
	FLSKIP = int_input('Enter the number of frames between fluorescence images (i.e. every nth image; for no fluorescence enter 0): ')	
	
if b_ALIGN or b_Segment or b_track or b_ANALYZE or b_RENDER:
	FNAME = False
	while not FNAME:
		fname = text_input('Enter the images\' filename preceding the channel and filenumber (e.g. 20171212_book): ')
		ftest = ImageDir + '/' + fname + '-%s-%03d.tif'
		FNAME = os.path.isfile(ftest %('p', FIRSTFRAME))
		if not FNAME:
			print 'cannot find file: ', ftest %('p', FIRSTFRAME)
			continue
		if (b_ALIGN or b_track or b_ANALYZE) and FLINITIAL != 0:
			FNAME = os.path.isfile(ftest %('g', FLINITIAL))
			if not FNAME:
				print 'cannot find file: ', ftest %('g', FLINITIAL)
else:
	ImageDir = None

if b_ANALYZE or b_track or b_RENDER:
	if FLINITIAL == 0:
		FLChannels = 0
	else:
		FLChannels = 1
	FLLABELS = []
	for i in range(FLChannels):
		label = text_input('Enter the name of fluorescence channel ' + str(i+1) +' (e.g. GFP): ')
		FLLABELS.append(label)

if b_ALIGN:
	ROIALIGNFILE = bool_input('Do you have a ROI file for a stationary area (Y/N): ')
	if ROIALIGNFILE:
		AlignROI = getfilename('Enter the path to the csv file, relative to the working directory (e.g. Align_roi.csv): ')
	else:
		AlignROI = 'None'
	AlignDir = text_input('Enter the name of the directory to output images into (e.g. Aligned): ')
else:
	AlignDir = ImageDir
		
if not b_Segment and (b_track or b_ANALYZE or b_RENDER):
	GETMASKS = bool_input('Do you have masks for the images? (Y/N): ')
	if GETMASKS:
		MASKDIR = False
		while not MASKDIR:
			Mask2Dir = text_input('Enter the name of the directory containing masks, relative to ' + AlignDir + ': ')
			MASKDIR = os.path.isdir(AlignDir + '/' + Mask2Dir)
			if not MASKDIR:
				print 'could not find directory'
	else:
		Mask2Dir = 'None'			

if b_ALIGN:
	AlignARG = [AlignROI, AlignDir, ImageDir, fname, FLINITIAL, FLSKIP]
	Image_alignment.run(AlignARG)


if b_Segment:
	IJPATH = False
	while not IJPATH:
		IMAGEJ = text_input('Enter the absolute path to the Fiji executable file (e.g. /home/user/Downloads/Fiji.app/ImageJ-linux64): ')
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
			openFiji = bool_input('Is there an open instance of Fiji? (Y/N): ')
			if not openFiji:
				print 'Please open Fiji'
		RunWeka.training(WekaARG1)
	
	ROUND2 = int_input('How many round of classification are you running? (1 or 2): ')
	if 'linux' in IMAGEJ:
		PROCESS = bool_input('Would you like to batch classify the images in the background? (Y/N): ')
	else:
		PROCESS = False
	if PROCESS:
		CORES = int_input('Enter how many processes are available to use for multiprocessing; set to 1 for no multiprocessing: ')
	else:
		CORES = None
	if ROUND2 == 2:
		
		TRAINING = bool_input('Do you have a trained classifier? (Y/N): ')
		if not TRAINING:
			#train first classifier
			training()
		
		#first batch classification
		classifierfile1 = getfilename('Enter the path to the classifier, relative to the working directory (e.g. Aligned/classifier.model): ')
		Mask1Dir = text_input('Enter the name of the directory within ' + AlignDir + ' to output masks into (e.g. Mask1): ')
		runMask1Dir = AlignDir + '/' + Mask1Dir
		
		if not os.path.isdir(runMask1Dir):
			os.system('mkdir ' + runMask1Dir)
		else:
			os.system('rm ' + runMask1Dir + '/' + fname + '*.png')
		WekaARG2 = [IMAGEJ, True, WorkDir + '/', AlignDir+ '/', runMask1Dir + '/', FIRSTFRAME, FRAMEMAX, fname + '-p', 'tif', classifierfile1, CORES] 
		RunWeka.batchsegment(WekaARG2)
		
		print 'continuing to second round of classification\n'
	else:
		runMask1Dir = getdirname('Enter the name of the directory containing images to classify, relative to the working directory (e.g. Aligned/Mask1): ')
		
	TRAINING2 = bool_input('Do you have a second trained classifier? (Y/N): ')
	if not TRAINING2:
		#train first classifier
		training()
	
	#second batch classification
	classifierfile2 = getfilename('Enter the path to the classifier, relative to the working directory (e.g. Aligned/classifier2.model): ')
	Mask2Dir = text_input('Enter the name of the directory within ' + AlignDir + ' to output masks into (e.g. Mask2): ')
	runMask2Dir = AlignDir + '/' + Mask2Dir
	
	if not os.path.isdir(runMask2Dir):
		os.system('mkdir ' + runMask2Dir)
	else:
		os.system('rm ' + runMask2Dir + '/' + fname + '*.png')
	WekaARG2 = [IMAGEJ, False, WorkDir + '/', runMask1Dir+ '/', runMask2Dir + '/', FIRSTFRAME, FRAMEMAX, fname + '-p', 'png', classifierfile2, CORES] 
	RunWeka.batchsegment(WekaARG2)

	
if b_track:
	if not Mask2Dir == 'None':
		LineageDir = 'Lineages'
		if not os.path.isdir(AlignDir + '/' + LineageDir):
			os.system('mkdir ' + AlignDir + '/' + LineageDir)
		AREAMIN = int_input('Enter the minimum cell area for tracking (e.g. 100): ')
		AREAMAX = int_input('Enter the maximum cell area for tracking (e.g. 2500): ')
		MINTRAJLENGTH = int_input('Enter the minimum number of frames to track cells through (e.g. 15): ')
		if MINTRAJLENGTH >= FRAMEMAX - 2:
			MINTRAJLENGTH = FRAMEMAX - 2
			print 'Maximum length is 2 less the total number of frames.'
			print 'Tracking through at least ', MINTRAJLENGTH, ' frames.'
	else:
		b_track = False
		print ('Tracking cells requires a mask directory.')

if b_ANALYZE or b_RENDER:
	ROIANALYZE = bool_input('Do you wish to analyze a region of interest? (Y/N): ')
	CroptoROI = bool_input('Do you wish to crop the images based on an ROI? (Y/N): ')
	if ROIANALYZE or CroptoROI:
		ROIFILE = getfilename('Enter the path to the ROI file to analyze, relative to the working directory (e.g. ROI.csv): ')
	else:
		ROIFILE = 'None'
	Writelineagetext = bool_input('Do you want to number the cells in the images based on lineage tracking? (Y/N): ')
	if not Mask2Dir == 'None':
		ContourImage = bool_input('Do you want to outline cells based on masks? (Y/N): ')
	else:
		ContourImage = False
		
if b_track:
	TrackARG = [AlignDir, fname, Mask2Dir, LineageDir, FIRSTFRAME, FRAMEMAX, AREAMIN, AREAMAX,FLLABELS, Ftime, FLSKIP, FLINITIAL,MINTRAJLENGTH]
	TrackCellLineages.run(TrackARG)
	
if LANALYZE:
	Lineage_analysis.run()
	
if b_ANALYZE or b_RENDER:
	if Writelineagetext:
		WRITETEXT = os.path.isfile('lineagetext.pkl')
		if not WRITETEXT:
			print 'Cannot find lineagetext.pkl to use for writing text.'
			Writelineagetext = False

	if FLINITIAL == 0:
		AnalyzeARG = [AlignDir, 1, b_ANALYZE, ROIFILE, b_RENDER, ContourImage, AlignDir + '/' + Mask2Dir, CroptoROI, Writelineagetext, FLChannels, FIRSTFRAME, fname, Ftime, FLLABELS]
	else:
		AnalyzeARG = [AlignDir, FLSKIP, b_ANALYZE, ROIFILE, b_RENDER, ContourImage, AlignDir + '/' + Mask2Dir, CroptoROI, Writelineagetext, FLChannels, FLINITIAL, fname, Ftime, FLLABELS]
	Image_analysis.run(AnalyzeARG)


