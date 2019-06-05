
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
		str = input(DISPLAY)	
	return str
	
# helper function for input
def bool_input(DISPLAY):
	bValue = None
	while (bValue is None):
		str = input(DISPLAY)
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
#run when training classifier
def training(argv):
	IMAGEJ = argv[0]
	homedir = argv[1]
	IJMSCRIPT = 'Segmentation.ijm'
	IJMSCRIPT = homedir + IJMSCRIPT
	#run ImageJ macro to train classifier
	OPEN_FMT = IMAGEJ + ' -port2 1> /dev/null 2> /dev/null &'
	print(OPEN_FMT)
	print('opening ImageJ')
	MACRO_FMT = IMAGEJ + ' --console -macro ' + IJMSCRIPT + ' -port2'
	os.system(OPEN_FMT)
	time.sleep(5)
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

	iXY = argv[11]
	
	#make sure number of xy regions won't overload computer
	if CORES == None or (len(iXY)>=CORES):
		PROCESS = False
		CORES = 2
		for ixy in iXY:
			COMMAND_FMT = IMAGEJ + ' --console ' + BSHSCRIPT + ' %d %d %d ' + Useprobs + ' ' + homedir + ' ' + imgdir + ' ' + segdir + ' ' + classifierfile + ' ' + fimageInFmt + ' ' + ext + ' ' + str(ixy)
			cmd = COMMAND_FMT % (FRAMEMIN, FRAMEMAX, FRAMESKIP)
			print(cmd)
			os.system(cmd)

	else:
		PROCESS = True
		for ixy in iXY:
			COMMAND_FMT = IMAGEJ + ' --console ' + BSHSCRIPT + ' %d %d %d ' + Useprobs + ' ' + homedir + ' ' + imgdir + ' ' + segdir + ' ' + classifierfile + ' ' + fimageInFmt + ' ' + ext  + ' ' + str(ixy) + ' 1> /dev/null 2> /dev/null &'
			for i in range(int((CORES-1)/len(iXY))):
				cmd = COMMAND_FMT % (FRAMEMIN+i, FRAMEMAX, FRAMESKIP*((CORES-1)/len(iXY)))
				os.system(cmd)
				#print cmd

	#COMMAND_FMT = '~/media/shared/drive/programs/Fiji.app/ImageJ-linux64 --headless --console single_image_classify_scale_arg.bsh %d %d %d' # 1> /dev/null 2> /dev/null &'


	#advanced option for running on a number of cores
	#number of cores you would like to use
	

	#check number of files in output directory until processes are complete
	i = 0
	CURSOR_UP_ONE = '\x1b[1A'
	#imgfiles = glob.glob(imgdir + '/' + fimageInFmt + '*')
	imgfiles = (FRAMEMAX + 1 - FRAMEMIN) *len(iXY)
	while PROCESS:
		i += 1
		time.sleep(1)
		maskfiles = glob.glob(segdir + '/*.png')
		if len(maskfiles) == imgfiles:
			PROCESS = False
		else:
			images = imgfiles - len(maskfiles)
			if i % 2 == 1:
				print('\\ images remaining: %d                 ' %images)
				sys.stdout.write(CURSOR_UP_ONE) 
			else:
				print('/ images remaining: %d                   ' %images)
				sys.stdout.write(CURSOR_UP_ONE) 
	print('finished classification                    ')
		
############################################################################
############################################################################

if __name__ == "__main__":
	print(sys.argv)
	if len(sys.argv) == 1:
		print('Please include additional argument /n training: if training a classifier n/ batch: if applying a classifier')
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
					print('Please open Fiji')
			WekaARG1 = [IMAGEJ, WorkDir + '/']
			training(WekaARG1)
		else:
			#Directory containing the images
			ImageDir = 'Test/'
			AlignDir = ImageDir
			
			#Image filename preceding channel indication (e.g. 20171212_book)
			fname = 't'
			
			#first frame of images
			FIRSTFRAME = 100
			#last frame of images
			FRAMEMAX = 120
			
			#name of Directory to output Masks made relative to image directory
			Mask1Dir = 'Mask1/'
			runMask1Dir = AlignDir + '/' + Mask1Dir
			
			#classifier file relative to working directory
			classifierfile1 = 'classifier.model'
			#classifierfile1 = getfilename('Enter path to classifier relative to working directory (e.g. Aligned/classifier.model): ')
			
			#True if producting probablity masks; False for binary masks
			Useprobability = True
			
			ext = 'tif'
			
			cores = 20
			
			
			iXY = [1]

			if not os.path.isdir(runMask1Dir):
				os.system('mkdir ' + runMask1Dir)
			else:
				os.system('rm ' + runMask1Dir + '/' + fname + '*.png')

				WekaARG2 = [IMAGEJ, Useprobability, WorkDir + '/', AlignDir , runMask1Dir + '/', FIRSTFRAME, FRAMEMAX, fname, ext, classifierfile1, cores, iXY] 
				batchsegment(WekaARG2)

			











