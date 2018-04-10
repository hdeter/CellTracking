
import sys
import glob
import os
import time


############################################################################
############################################################################
	
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

#function for getting and checking filenames
def getfilename(prompt):
	ISFILE = False
	while not ISFILE:
		FILENAME = text_input(prompt)
		ISFILE = os.path.isfile(FILENAME)
	return FILENAME
############################################################################
############################################################################
#run when training classifier
def training(argv):
	IMAGEJ = argv[0]
	homedir = argv[1]
	IJMSCRIPT = 'Segmentation.ijm'
	IJMSCRIPT = homedir + IJMSCRIPT
	#run ImageJ macro to train classifier
	MACRO_FMT = IMAGEJ + ' --console -macro ' + IJMSCRIPT
	os.system(MACRO_FMT)

############################################################################
############################################################################
#run when classifying a batch of images
def batchsegment(argv):
		
	#location of executable ImageJ file (from root)
	IMAGEJ = argv[0]

	#working directory (the directory the script is in)
	homedir = argv[2]


	#parameters for training a classifier
	#path to ijm script relatvie to working directory


	#parameters for batch segmentation

	#path to beanshell script relative to working directory
	BSHSCRIPT = 'Batch_segment.bsh'

	#set to str(0) if classifying phase image to probability mask
	#set to str(1) if classifying probability mask to binary mask
	Useprobability = argv[1]
	if Useprobability:
		Useprobs = str(0)
	else:
		Useprobs = str(1)

	#path to directory containing images to be classified relative to working directory
	imgdir = argv[3]
	#path to directory to save masks in relative to working directory
	segdir = argv[4]
	#path to clasifier relative to working directory
	#classifierfile = 'Aligned/031918-classifier.model'
	classifierfile = argv[9]
	#Name of images preceding filenumber
	fimageInFmt = argv[7]
	#Filetype of images being classified (tif or png)
	ext = argv[8]
	#ext = 'png'

	#First frame to be classified
	FRAMEMIN = argv[5]
	#Last frame to be classified
	FRAMEMAX = argv[6]
	#Classify every nth frame (e.g. to classify every 10 frames, set to 10)
	FRAMESKIP =1
	CORES = argv[10]

	if CORES == None:
		PROCESS = False
		CORES = 2
		COMMAND_FMT = IMAGEJ + ' --console ' + BSHSCRIPT + ' %d %d %d ' + Useprobs + ' ' + homedir + ' ' + imgdir + ' ' + segdir + ' ' + classifierfile + ' ' + fimageInFmt + ' ' + ext 
		cmd = COMMAND_FMT % (FRAMEMIN, FRAMEMAX, FRAMESKIP)
		os.system(cmd)

	else:
		PROCESS = True
		COMMAND_FMT = IMAGEJ + ' --console ' + BSHSCRIPT + ' %d %d %d ' + Useprobs + ' ' + homedir + ' ' + imgdir + ' ' + segdir + ' ' + classifierfile + ' ' + fimageInFmt + ' ' + ext + ' 1> /dev/null 2> /dev/null &'
		for i in range(CORES-1):
			cmd = COMMAND_FMT % (FRAMEMIN+i, FRAMEMAX, FRAMESKIP*(CORES-1))
			os.system(cmd)
			#print cmd

	#COMMAND_FMT = '~/media/shared/drive/programs/Fiji.app/ImageJ-linux64 --headless --console single_image_classify_scale_arg.bsh %d %d %d' # 1> /dev/null 2> /dev/null &'


	#advanced option for running on a number of cores
	#number of cores you would like to use
	



	#check number of files in output directory until processes are complete
	i = 0
	CURSOR_UP_ONE = '\x1b[1A'
	imgfiles = glob.glob(imgdir + '/' + fimageInFmt + '*')
	while PROCESS:
		i += 1
		time.sleep(1)
		maskfiles = os.listdir(segdir)
		if len(maskfiles) == len(imgfiles):
			PROCESS = False
		else:
			images = len(imgfiles) - len(maskfiles)
			if i % 2 == 1:
				print '\\ classifying %03d images' %images
				sys.stdout.write(CURSOR_UP_ONE) 
			else:
				print '/ classifying %03d images' %images
				sys.stdout.write(CURSOR_UP_ONE) 
	print 'finished classification                    '
		
############################################################################
############################################################################

if __name__ == "__main__":
	print sys.argv
	if len(sys.argv) == 1:
		print 'Please include additional argument /n training: if training a classifier n/ batch: if applying a classifier'
	elif len(sys.argv) > 2:
		main(sys.argv[1:])
	else:
		argv = sys.argv
		train = argv[1]
		#location of ImageJ executable file
		IMAGEJ = '/media/shared/drive/programs/newFiji/Fiji.app/ImageJ-linux64'
		#get working directory
		WorkDir = os.getcwd()

		if train == 'training':
			openFiji = False
			while not openFiji:
				openFiji = bool_input('Is there an open instance of Fiji (Y/N): ')
				if not openFiji:
					print 'Please open Fiji'
			WekaARG1 = [IMAGEJ, WorkDir + '/']
			training(WekaARG1)
		else:
			#Directory containing the images
			ImageDir = 'Test2'
			AlignDir = ImageDir
			
			#Image filename preceding channel indication (e.g. 20171212_book)
			fname = '20171212_book'
			
			#first frame of images
			FIRSTFRAME = 448
			#last frame of images
			FRAMEMAX = 467
			
			#name of Directory to output Masks made relative to image directory
			Mask1Dir = 'Mask2'
			runMask1Dir = AlignDir + '/' + Mask1Dir
			
			#classifier file relative to working directory
			classifierfile1 = 'Aligned/classifier2.model'
			#classifierfile1 = getfilename('Enter path to classifier relative to working directory (e.g. Aligned/classifier.model): ')
			
			#True if producting probablity masks; False for binary masks
			Useprobability = False
			
			ext = 'png'
			
			cores = 21
			
			
			if not os.path.isfile(runMask1Dir):
				os.system('mkdir ' + runMask1Dir)
			else:
			os.system('rm ' + runMask1Dir + '/' + fname + '*.' + ext)
			
			WekaARG2 = [IMAGEJ, Useprobability, WorkDir + '/', AlignDir+ '/Mask1/', runMask1Dir + '/', FIRSTFRAME, FRAMEMAX, fname + '-p', ext, classifierfile1, cores] 
			batchsegment(WekaARG2)

			











