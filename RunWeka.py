#note that an instance of Fiji must be open to access the toolbar when training a classifier
import sys
import os

############################################################################
############################################################################

#location of executable ImageJ file (from root)
IMAGEJ = '/Users/ellis/Downloads/Fiji.app/Contents/MacOS/ImageJ-macosx'
homedir = '/Users/ellis/Downloads/DeterZip20/'

#set to true if training a classifier
training = True


if training:
	IJMSCRIPT = 'Segmentation.ijm'
	#run ImageJ macro to train classifier
	MACRO_FMT = IMAGEJ + ' --console -macro ' + homedir + IJMSCRIPT
	os.system(MACRO_FMT)

else:
	BSHSCRIPT = 'Batch_segment.bsh'
	#set to 0 for True and 1 for False
	Useprobs = str(1)

	imgdir = 'Aligned/Mask1/'
	segdir = 'Aligned/Mask2/'
	classifierfile = 'Aligned/Mask1/classifier2.model'
	fimageInFmt = '20171212_book-p-'
	ext = 'png'
	FRAMEMIN = 448
	FRAMEMAX = 467
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















