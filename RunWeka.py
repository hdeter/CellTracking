
import sys
import os
import time

############################################################################
############################################################################

#location of executable ImageJ file (from root)
IMAGEJ = '/media/shared/drive/programs/oldFiji/Fiji.app/ImageJ-linux64'

#working directory (the directory the script is in)
homedir = '/media/hdeter/drive/DiesMiracle/Book_chapter/DeterZip20/'
#set to true if training a classifier
training = True

#parameters for training a classifier
#path to ijm script relatvie to working directory
IJMSCRIPT = 'Segmentation.ijm'

#parameters for batch segmentation

#path to beanshell script relative to working directory
BSHSCRIPT = 'Batch_segment.bsh'

#set to str(0) if classifying phase image to probability mask
#set to str(1) if classifying probability mask to binary mask
#Useprobs = str(0)
Useprobs = str(1)

#path to directory containing images to be classified relative to working directory
imgdir = 'Aligned/Mask1/'
#path to directory to save masks in relative to working directory
segdir = 'Aligned/Mask2/'
#path to clasifier relative to working directory
#classifierfile = 'Aligned/031918-classifier.model'
classifierfile = 'Aligned/Mask1/013118_classifier2.model'
#Name of images preceding filenumber
fimageInFmt = '20171212_book-p'
#Filetype of images being classified (tif or png)
ext = 'tif'
#ext = 'png'

#First frame to be classified
FRAMEMIN = 448
#Last frame to be classified
FRAMEMAX = 467
#Classify every nth frame (e.g. to classify every 10 frames, set to 10)
FRAMESKIP =1

if training:

	IJMSCRIPT = homedir + IJMSCRIPT
	#run ImageJ macro to train classifier
	MACRO_FMT = IMAGEJ + ' --console -macro ' + IJMSCRIPT
	os.system('gnome-terminal -e \'' + IMAGEJ + '\'')
	time.sleep(6)
	os.system(MACRO_FMT)

else:

	COMMAND_FMT = IMAGEJ + ' --console ' + BSHSCRIPT + ' %d %d %d ' + Useprobs + ' ' + homedir + ' ' + imgdir + ' ' + segdir + ' ' + classifierfile + ' ' + fimageInFmt + ' ' + ext
	cmd = COMMAND_FMT % (FRAMEMIN, FRAMEMAX, FRAMESKIP)
	os.system(cmd)

	#COMMAND_FMT = '~/media/shared/drive/programs/Fiji.app/ImageJ-linux64 --headless --console single_image_classify_scale_arg.bsh %d %d %d' # 1> /dev/null 2> /dev/null &'


	#advanced option for running on a number of cores
	#number of cores you would like to use
	#~ CORES = 8

	#~ for i in range(CORES):
		#~ cmd = COMMAND_FMT % (FRAMEMIN+i, FRAMEMAX, FRAMESKIP*CORES)
		#~ os.system(cmd)
		#~ print cmd















