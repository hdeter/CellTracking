
import sys
import os
import time

############################################################################
############################################################################

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
	return FILENAME
	
def getdirname(prompt):
	ISPATH = False
	while not ISPATH:
		PATHNAME = text_input(prompt)
		ISPATH = os.path.isdir(PATHNAME)
	return PATHNAME

####################################################################
####################################################################
#########################################################################
########################################################################
#location of executable ImageJ file (from root)
#IMAGEJ = '/media/shared/drive/programs/oldFiji/Fiji.app/ImageJ-linux64'
IMAGEJ = text_input('Add the path to ImageJ-linux64 relative to root directory (e.g. /home/user/DeterZip20/Fiji.app/ImageJ-linux64): ')
homedir = getdirname('Enter working directory (e.g. /home/user/DeterZip20): ')
homedir = homedir + '/'
#set to true if training a classifier
training = bool_input('Are you training a classifier (Y/N): ')


if training:
	#IJMSCRIPT = 'Segmentation.ijm'
	IJMSCRIPT = getfilename('Enter path to Segmentation.ijm relative to working directory (e.g. Segmentation.ijm): ')
	#run ImageJ macro to train classifier
	MACRO_FMT = IMAGEJ + ' --console -macro ' + IJMSCRIPT
		os.system('gnome-terminal -e \'' + IMAGEJ + '\'')
	time.sleep(6)
	os.system(MACRO_FMT)


else:
	
	#BSHSCRIPT = 'Weka_segment_script.bsh'
	BSHSCRIPT = getfilename('Enter path to Batch_segment.bsh relative to working directory (e.g. Batch_segment.bsh): ')
	
	#set to 0 for True and 1 for False
	Useprobability = bool_input('Are you classifying a phase image (Y/N): ')
	if Useprobability:
		Useprobs = str(0)
	else: 
		Useprobs = str(1)

	imgdir = getdirname('Enter path to images relative to working directory (e.g. Aligned): ')
	imgdir = imgdir + '/'
	segdir = text_input('Enter path to directory for segmented images relative to working directory (e.g. Aligned/Mask1): ')
	segdir = segdir + '/'
	if not os.path.isfile(segdir):
		os.system('mkdir ' + segdir)
	classifierfile = getfilename('Enter path to classifier relative to working directory (e.g. Aligned/031918-classifier.model): ')
	imageFmtB = False
	FRAMEMIN = int_input('Enter the number of the first frame to classify (e.g. 448): ')
	FRAMEMAX = int_input('Enter the number of the last frame to classify (e.g. 467): ')
	while not imageFmtB:
		fimageInFmt = text_input('Enter image filenames preceding file numbers (relative to image directory; e.g. 20171212_book-p): ')
		ext = text_input('Enter extension of image files (e.g. tif, png): ')
		imgFmt = imgdir + fimageInFmt + '-%03d.' + ext
		imgfile = imgFmt %FRAMEMIN
		imageFmtB = os.path.isfile(imgfile)
		if not imageFmtB:
			print 'unable to find file: ', imgfile, ' in image directory (' + imgdir + ')'

	COMMAND_FMT = IMAGEJ + ' --console ' + BSHSCRIPT + ' %d %d %d ' + Useprobs + ' ' + homedir + ' ' + imgdir + ' ' + segdir + ' ' + classifierfile + ' ' + fimageInFmt + ' ' + ext
	cmd = COMMAND_FMT % (FRAMEMIN, FRAMEMAX, 1)
	os.system(cmd)

#COMMAND_FMT = '~/media/shared/drive/programs/Fiji.app/ImageJ-linux64 --headless --console single_image_classify_scale_arg.bsh %d %d %d' # 1> /dev/null 2> /dev/null &'



#advanced option for running on a number of cores
#number of cores you would like to use
#CORES = 8

#~ for i in range(CORES):
	#~ cmd = COMMAND_FMT % (FRAMEMIN+i, FRAMEMAX, FRAMESKIP*CORES)
	#~ os.system(cmd)
	#~ print cmd
















