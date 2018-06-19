#the purpose of this script is to use binary masks to label and track cells
#some modifications are required for the script to work within a given directory and with a specific dataset
############################################################################
############################################################################

from time import sleep
from time import clock

import numpy as np
import numpy.random as rnd

import pdb

import string

import sys
import os

import cPickle as pickle

from multiprocessing import Pool

import scipy.ndimage as ndimage
import scipy.misc as misc

import scipy.sparse

import cv2 as cv
from StringIO import StringIO

from scipy import convolve

############################################################################
#globals
#used to keep track of which frames have fluorescence images
FLFILES = []

############################################################################
############################################################################

# filter 1: filter to set minimum to zero and standard deviation to 1
def filterData1(data):
    data = data - np.min(data)
    data = data / np.std(data)
    return data*1.0

# sets the filtering window size for filter2 (10 is a good standard number)
FILTERWINDOW = 10

# filter 2: high pass filter, subtracts running average of 10 frames, normalizes standard deviation to 1
# in the future - make these easy to pass as parameters
def filterData2(data):
	#pdb.set_trace()
	WIND = FILTERWINDOW
	data = data-convolve(data, np.ones((WIND))/WIND, mode='same')
	#data = data-np.min(data[50:100])
	
	data[0:WIND] = 0.0
	data[-WIND:] = 0.0
	
	#data = data / (np.std(data[50:300]))
	data = data / (np.std(data))

	return data

# filter 3: filter to set the minimum to zero and maximum to 1
def filterData3(data):
    data = data - np.min(data)
    data = data / np.max(data)
    return data*1.0

def backgroundData2(data):
	#pdb.set_trace()
	WIND = FILTERWINDOW
	
	dataBackground = convolve(data, np.ones((WIND))/WIND, mode='same')
	
	return dataBackground
	

# add grayscale to color image
def addGrayToColor(IMG,img):
    for ii in range(3):
        IMG[:,:,ii] = IMG[:,:,ii] + img
    return IMG
    
############################################################################
############################################################################

#open the color table to get colors for stuff
#need to have colorfile in working directory
def getColors():
	colorfile = 'RGB-codes.csv'
	ISFILE = os.path.isfile(colorfile)
	if not ISFILE:
		#colors = [(65025,0,0),(0,0,65025),(65025,65025,0),(0,65025,65025),(65025,0,65025)]
		colors = [(255,0,0),(0,0,255),(255,255,0),(0,255,255),(255,0,255)]
	else:
		#print 'TRYING TO IMPORT ROI:', ROIFILE
		COLORTXT = open(colorfile, 'rb').read()

		# trick for newline problems
		COLORTXT = StringIO(COLORTXT.replace('\r','\n\r'))
		colors = []
		COLORS = np.loadtxt(COLORTXT, delimiter=",",skiprows=1, dtype='str')
		for row in COLORS:
			color = (int(row[1]),int(row[2]),int(row[3]))
			colors.append(color)

	return colors[2:]



############################################################################
############################################################################
#parameters to adjust based on directory and dataset
def main(argv):
	#the path to the directory containing original/aligned images relative working directory
	rootdir = argv[0] + '/'

	fname = argv[1]
	MaskDir = argv[2] + '/'
	
	#the following are added to the rootdir to get specific files
	#the filenames should be numbered (in this case with three decimals; %03d)
	#mask image
	filefmt = MaskDir + fname + '-p-%03d.png'
	#phase image
	filefmt2 = fname + '-p-%03d.tif'
	
	#useful parameters for if you want to visualize cell and compare to output
	#set to true to create labeled images with numbers (trajectories)
	writeNumbers = True
	#set to true to create labeled images with cell regions false colored
	writeLabels = True
	#image to write labels on (currently set to mask images)
	labelimage = filefmt
	#name to save labeled stuff
	labelname = argv[3] + '/' + fname + '-%03d.png'


	#output name for cell statistics csv
	cellstatCSV = 'global_cell_statistics.csv'
	#output name for lineages csv
	lineageCSV = 'lineagedata.csv'

	# fluorescence channels (used in filenames)
	#used to get string in filename below. can be modified to use decimals rather than strings
	iCF = ['g']
	#fluorescence image
	filefmtfl = fname + '-%s-%03d.tif'

	#the number for the first frame in the dataset
	FIRSTFRAME = argv[4]
	#used to test every nth image for overlap.
	FRAMESKIP = 1
	#the number of the last frame in the dataset
	FRAMEMAX = argv[5]
	
	# limits on the area of cells to consider
	AREAMIN = argv[6]
	AREAMAX = argv[7]
	
	# radius of how far center of masses can be and still check distances
	THRESHOLD = AREAMIN*1.5

	#used in csv output
	FLLABELS = argv[8]

	# time in between analyzed frames (different from frames)
	dt = argv[9]  # min

	# period of fluorescence frames (every nth frame)
	dPeriodFL = argv[10]  # frames
	#the first frame that has a fluorescence image
	FLINITIAL = argv[11]

	# minimum trajectory length (in actual segmented frames)
	MINTRAJLENGTH = argv[12] #frames

	#end of parameters to adjust
	############################################################################
	############################################################################


	############################################################################
	############################################################################
	def getLabeledMask(iFRAME):
		
		global FLFILES

		# get segmentation data
		filename = rootdir + filefmt % (iFRAME)
		img = ndimage.imread(filename)
		#print 'loaded label file', iFRAME, ' with dimensions', img.shape
		img = 255-img
		
		# get fluorescence data
		imgFLALL = []
		for icf in iCF:
			try:
				filename = rootdir + filefmtfl % (icf, iFRAME)
				#print filename
				imgFL = ndimage.imread(filename)
				FLFILES.append(iFRAME)
			except:
				imgFL = None

			
			#print 'loaded fluorescence file', iFRAME, ' with dimensions', imgFL.shape
			imgFLALL.append(imgFL)
			

		# unpack segmentation data, after labeling
		label, nlabels = ndimage.label(img)

		return (label, nlabels, imgFLALL)


	############################################################################
	############################################################################

	# gets center of mass (CoM) and area for each object
	def getObjectStats(label, nlabels, FLALL, i, iFRAME):
		# measure center of mass and area
		comXY = ndimage.center_of_mass(label*0 + 1.0, label, range(nlabels))
		AREA = ndimage.sum(label*0 + 1.0, label, range(nlabels))
		
		# measure mean fluorescence
		FLMEASURE = []

		if iFRAME in FLFILES:
			for img in FLALL:
				flint = ndimage.sum(img, label, range(nlabels))
				flmean = flint / AREA
				FLMEASURE.append(flmean)
		else:
			FLMEASURE.append(None)
			
		if i == 1:
			labelsum = ndimage.sum(label, label, range(nlabels))
			celllabels = labelsum / AREA
			
			return (comXY, AREA, celllabels, FLMEASURE)
		else:
			return (comXY, AREA, FLMEASURE)
		
		
		
	############################################################################
	############################################################################


	# for a given index (i2), compute overlap with all i1 indices
	def getMetricOverlap(ARG):
		
		i2,label1,label2,nlabels1,nlabels2,comXY1,comXY2 = ARG

		# optimized fastest method?
		
		# get a subregion of the label image, then compare
		SZ = label1.shape
		Xlow = max([0,int(comXY2[0,i2]-THRESHOLD)])
		Ylow = max([0,int(comXY2[1,i2]-THRESHOLD)])
		Xhigh = min([SZ[0],int(comXY2[0,i2]+THRESHOLD)])
		Yhigh = min([SZ[1],int(comXY2[1,i2]+THRESHOLD)])

		# i2c = label2[int(comXY2[0,i2]),int(comXY2[1,i2])]
		# if (i2c == i2):
		#	print i2, i2c

		label1 = label1[Xlow:Xhigh,Ylow:Yhigh]
		label2 = label2[Xlow:Xhigh,Ylow:Yhigh]
		
		# print i2, SZ, comXY2[:,i2], Xlow, Xhigh,Ylow, Yhigh

		# finally, compute overlap using a simple function
		overlap = ndimage.sum(label2==i2, label1, range(nlabels1))
		#notzero = np.nonzero(overlap)
		#print notzero, i2

		#overlap0 = ndimage.sum(label2==i2, label2, [i2,])
		#if (np.max(overlap)/np.max(overlap0) > 1.0):
		#	print np.max(overlap)/np.max(overlap0)

		return overlap


	############################################################################
	############################################################################


	def runOnce(iFRAME):
		print 'frame = ', iFRAME
		
		# make labels from the masks
		label1, nlabels1, FL1ALL = getLabeledMask(iFRAME)
		label2, nlabels2, FL2ALL = getLabeledMask(iFRAME+FRAMESKIP)
		
		# get statistics
		comXY1, AREA1, celllabels1, FLMEASURE1 = getObjectStats(label1, nlabels1, FL1ALL, 1, iFRAME)
		comXY2, AREA2, FLMEASURE2 = getObjectStats(label2, nlabels2, FL2ALL, 2, iFRAME+FRAMESKIP)
		
		# process center of mass (CoM) arrays
		comXY1 = np.array(zip(*comXY1))
		comXY2 = np.array(zip(*comXY2))
		comXY1 = np.nan_to_num(comXY1)
		comXY2 = np.nan_to_num(comXY2)
		
		# make the distance map
		DISTMAT = []
		ARGALL = []
		for i2 in range(nlabels2):
			ARGALL.append((i2,label1,label2,nlabels1,nlabels2,comXY1,comXY2))
		DISTMAT = map(getMetricOverlap, ARGALL)
		DISTMAT = np.array(DISTMAT)
		#print DISTMAT.shape
		
		# return a compressed sparse array, since most entries are zeros
		DISTMAT = scipy.sparse.csr_matrix(DISTMAT)

		# print 'frame ', iFRAME, ' done'
		
		#pdb.set_trace()

		return (iFRAME, label1, nlabels1, (comXY1, celllabels1, AREA1), DISTMAT, FLMEASURE1)


	############################################################################
	############################################################################


	############################################################################
	############################################################################

	#track lineages proper
	#currently will only output first channel of fluorescence values in csv and pkl files

	#get all the positions and save them in nested dictionaries by frame then trajectory
	def getPositions(trajnum, XY, iFRAME, label):
		XY = int(XY[1]),int(XY[0])	
		LOC[iFRAME][trajnum] = XY
		LABELS[iFRAME][trajnum] = label

	#write to the image and save in images dictionary
	def writeLabel(newtrajnum, XY, label, iFRAME):
		img = images[iFRAME]
			
		color = colors[(int(newtrajnum[0:4]) % len(colors))]

		#print newtrajnum, iFRAME, XY, len(label)
		
		#could create a mask based on single label and combines it with image
		#right now it just edits the image itself
		#fnp = np.empty(img.shape)
		for j,k in label:
			#first line uses colors from colorfile, second line just uses red
			img[j,k] = color
			#img[j,k] = (255,0,0)
		
		#pdb.set_trace()
		#img = img + fnp*0.1

		images[iFRAME] = img

	#function that re-lables all trajectories by iFRAME then calls writeImage
	def relableFRAME(iFRAME):
		for loc in LOC[iFRAME]:
			XY = LOC[iFRAME][loc]
			label = LABELS[iFRAME][loc]

			if loc in NEWTRAJ:
				newtrajnum = NEWTRAJ[loc]
				
			else:
				DIVIDE[loc] = []
				for key in LOC[iFRAME]:
					#test if location matches that of a previous trajectory
					if (LOC[iFRAME][loc] == LOC[iFRAME][key]) and (key!= loc) and (key in NEWTRAJ):
						if key in BRANCH:
							num = BRANCH[key] +1
						else:
							num = 1
						BRANCH[key] = num
						newtrajnum = NEWTRAJ[key]
						newtrajnum2 = newtrajnum + '-%d' %num
						NEWTRAJ[loc] = newtrajnum2
						#add division time to both trajectories
						DIVIDE[key].append(iFRAME)
						DIVIDE[loc].append(iFRAME)
				if loc not in NEWTRAJ:
					cellvisit = len(NEWTRAJ) + 1
					newtrajnum = '%04d' %cellvisit
					NEWTRAJ[loc] = newtrajnum

			if writeLabels:
				writeLabel(newtrajnum, XY, label, iFRAME)

			#for writing label text in images
			towrite = newtrajnum, XY
			TOWRITE[iFRAME].append(towrite)
		

	#has to run after labels so they aren't written over	
	def writeText(iFRAME):
		img = images[iFRAME]
		for line in TOWRITE[iFRAME]:
			XY = line[1]
			newtrajnum = line[0]
			cv.putText(img, newtrajnum, XY, cv.FONT_HERSHEY_DUPLEX, .3, (0,0,0), 1)
		return img



	# get measurements in parallel
	#~ MEASUREMENTS = []
	#~ ARGLIST = []
	#~ for iFRAME in range(FIRSTFRAME,FRAMEMAX,1):
		#~ ARGLIST.append(iFRAME)

	#~ MEASUREMENTS = map(runOnce, ARGLIST)


	#~ #pdb.set_trace()

	#~ ############################################################################
	############################################################################

	#~ # save results
	#~ fpkl = open('trackMasks.pkl', 'wb')
	#~ pickle.dump(MEASUREMENTS, fpkl, protocol=pickle.HIGHEST_PROTOCOL)
	#~ fpkl.close()

	#~ ############################################################################
	#~ ############################################################################
	#analyzeTracking

	#~ with open('trackMasks.pkl', 'rb') as f:
		#~ TRACKING_RESULTS = pickle.load(f)
	
	#~ TRACKING_RESULTS = MEASUREMENTS
	#~ # unpack results into handy variables
	#~ iFRAME, label, nlabels, CELLSTATS, DISTMAT, FLMEASURE = zip(*TRACKING_RESULTS)

	#~ # cell statistics specifically
	#~ comXY, celllabels1, AREA = zip(*CELLSTATS)

	#~ # number of fluorescence channels
	#~ FLN = len(FLMEASURE[0])

	############################################################################
	############################################################################
	#~ #get total cell data
	#~ #get total cell data
	#~ # total number of frames
	#~ iTN = len(FLMEASURE)

	#~ # place to save mean and std data
	#~ FLMEANALL = []
	#~ FLMEANALLFILTERED = []
	#~ FLMEANALLBACKGROUND = []
	#~ FLSTDALL = []
	#~ FLMEDIANALL = []

	#~ # plot mean and std. dev. intensity vs. time across cells
	#~ for ifl in range(0,FLN):
		#~ print 'IFL =', ifl
		
		#~ fln = []
		#~ fltime = []
		#~ flmean = []
		#~ flstd = []
		#~ flmedian = []
		
		#~ # for iT in range(0,iTN,dPeriodFL):
		#~ for iT in range((FLINITIAL-FIRSTFRAME),(FRAMEMAX-FIRSTFRAME),dPeriodFL):
			#~ print iT, ifl
			#~ flframe = iT+FIRSTFRAME
			#~ #pdb.set_trace()
			#~ fl = np.array(FLMEASURE[iT][ifl])
			#~ area = np.array(AREA[iT])
			#~ iselect = np.where((area>=AREAMIN)*(area<=AREAMAX))[0]
			
			#~ fln.append(len(iselect))
			#~ fltime.append(dt*flframe)
			#~ flmean.append(np.mean(fl[iselect]))
			#~ flmedian.append(np.median(fl[iselect]))
			#~ flstd.append(np.std(fl[iselect]))

		#~ # save background for later use
		#~ #print 'flmean = ', flmean
		#~ FLMEANALL.append(flmean)
		#~ FLMEDIANALL.append(flmedian)
		#~ FLSTDALL.append(flstd)
		#~ FLMEANALLFILTERED.append(filterData1(flmean))
		#~ FLMEANALLBACKGROUND.append(backgroundData2(flmean))

	#~ with open('global-cell-statistics.pkl', 'wb') as f:
		#~ pickle.dump((fltime,fln,FLMEANALL,FLMEDIANALL,FLSTDALL,FLMEANALLFILTERED,FLMEANALLBACKGROUND), f, protocol=pickle.HIGHEST_PROTOCOL)
		


	#~ #save data to CSV
	#~ #need to loop through number of channels

	#~ f = open(cellstatCSV, 'wb')
	#~ f.write('time,')
	#~ f.write('cell count,')
	#~ for lmn in range(len(FLMEANALL)):
		#~ f.write('%s mean,' %FLLABELS[lmn])
		#~ f.write('%s std.,' %FLLABELS[lmn])
		#~ f.write('%s median,' %FLLABELS[lmn])
	#~ f.write('\n')

	#~ for ijk in range(len(fltime)):
		#~ f.write(str(fltime[ijk]) + ',')
		#~ f.write(str(fln[ijk]) + ',')
		#~ for lmn in range(len(FLMEANALL)):
			#~ f.write(str(FLMEANALL[lmn][ijk]) + ',')
			#~ f.write(str(FLSTDALL[lmn][ijk]) + ',')
			#~ f.write(str(FLMEDIANALL[lmn][ijk]) + ',')
		#~ f.write('\n')			

	#~ f.close()	
		
	#~ #pdb.set_trace()

	#~ # begin tracking proper

	#~ # find many trajectories by starting at a multitude of frames
	#~ FRAMEMAXLIST = range(FRAMEMAX,(FIRSTFRAME+MINTRAJLENGTH),-1)
	#~ # FRAMEMAXLIST = [FRAMEMAX,FRAMEMAX-30,FRAMEMAX-30*2,FRAMEMAX-30*3,FRAMEMAX-30*4]

	#~ # current number of trajectories
	#~ TRAJCOUNT = 0

	#~ # store trajectories
	#~ TRAJ = []

	#~ # keep track of which indices are visited
	#~ VISITED = []
	#~ for ijk in range(FRAMEMAX+1):
		#~ VISITED.append([])



	#~ # scan through the final frame
	#~ for FRAMEMAX in FRAMEMAXLIST:
		#~ print 'framemax ', FRAMEMAX

		#~ # scan all cell ID's at the final frame
		#~ cellIDStart = range(nlabels[FRAMEMAX-FIRSTFRAME-1])

		#~ # current cell ID
		#~ cellID = 0

		#~ # loop
		#~ for cellID in cellIDStart:
			#~ #print 'tracking cell ID: ', cellID

			#~ # frame
			#~ frame = []

			#~ # time
			#~ time = []

			#~ # area
			#~ area = []

			#~ # yfp
			#~ fl0 = []
			
			#~ if FLN == 2:
				#~ # cfp
				#~ fl1 = []
		
			#~ # flag for a "bad" trajectory
			#~ bBad = False
			
			#~ #for cell positions
			#~ cellXY = []
			
			#~ #for cell label
			#~ celllabels = []

			#~ loop = 0

			#~ for iT in range(FRAMEMAX-FIRSTFRAME-1, 0, -1):
			
				#~ # mark cell as visited if not visited, otherwise end trajectory
				#~ if (cellID in VISITED[iT]):
					#~ bBad = True
				#~ else:
					#~ VISITED[iT].append(cellID)
				

				#~ # find next best cell
				#~ dist = DISTMAT[iT-1]

				#~ # optionally print out shape information	
				#~ # print iT, cellID, dist.shape, nlabels[iT], nlabels[iT-1]


				#~ # area
				#~ area1 = AREA[iT][cellID]
				#~ if (not ( (area1>=AREAMIN) and (area1<=AREAMAX) )):
					#~ #print 'bad area ', area1, ' cell ', cellID
					#~ bBad = True
					
				#~ # area of potential matches
				#~ area2 = np.array(AREA[iT-1])
				
				#~ iselect2 = np.where((area2>=AREAMIN)*(area2<=AREAMAX))[0]
				#~ # print iselect2
		
				#~ dist2 = np.squeeze(dist[cellID,:].toarray())
				#~ dist2 = dist2[iselect2]
			
				#~ cellID2 = iselect2[np.argmax(dist2)]
				#~ distmax = np.amax(dist2)
			
				#~ # record current data
				
				#~ #print 'iT =', iT, 'cellID = ', cellID
				#~ #iFRAME if off from iT by 1 so add 1 to get frame
				#~ frame.append(iT+FIRSTFRAME)
					
				#~ CELLX = comXY[iT][0][cellID]
				#~ CELLY = comXY[iT][1][cellID]
				#~ CELLXY = [CELLX, CELLY]
				#~ #print cellXY
				#~ cellXY.append(CELLXY)
				
				#~ celllabel = celllabels1[iT][cellID]
				#~ framelabel = label[iT]
				#~ celllabel = np.where(framelabel == celllabel)
				#~ celllabel = np.column_stack(celllabel)
				#~ celllabels.append(celllabel)
				
				#~ time.append(dt*(iT+FIRSTFRAME))
				#~ area.append(area1)
				#~ if (((iT+FIRSTFRAME - FLINITIAL) % dPeriodFL == 0) and ((iT + FIRSTFRAME) >= FLINITIAL)):
					#~ try:
						#~ fl0.append(FLMEASURE[iT][0][cellID])
						#~ if FLN == 2:
							#~ fl1.append(FLMEASURE[iT][1][cellID])
					#~ except: 
						#~ print 'no fl data for ' + str(iT) 
				#~ else:
					#~ fl0.append(None)
					#~ if FLN == 2:
						#~ fl1.append(None)
					

				#~ # area in the previous time
				#~ area2 = AREA[iT-1][cellID2]


				#~ ####################
				#~ ####################

				#~ # only check if not bad
				#~ if (not bBad):
				
					#~ # check for wrong rate of change for area
					#~ if ((area2-area1)/area1 < -0.6):
						#~ bBad = True
						#~ #print 'cell ', cellID, ' failed due to area shrinkage = ', (area2-area1)/area1
						#~ #print '\tarea 1 = ', area1
						#~ #print '\tarea 2 = ', area2

					#~ # check for strong overlap
					#~ if ((distmax/area1) < 0.5):
						#~ bBad = True
						#~ #print 'cell ', cellID, ' failed due to low overlap = ', (distmax/area1)
						#~ #print '\tarea 1 = ', area1
						#~ #print '\tarea 2 = ', area2

				#~ if (bBad):
					#~ break
				

				#~ ####################
				#~ ####################

				#~ cellID = cellID2
		
				#~ # pdb.set_trace()
		

			#~ # if ((np.max(flarea)<800) and (not bBad)):
			#~ # if ((len(fltime)>100) and (np.max(flarea)<800)):
			#~ #pdb.set_trace()
			#~ #print 'fmeasures ', fltime, np.mean(flarea), np.max(fl0)
			
			#~ #if ((len(fltime)>MINTRAJLENGTH) and (np.mean(flarea)<500) and (np.max(fl0)<4000)):
			#~ if ((len(frame)>MINTRAJLENGTH)):				
				#~ if FLN == 2:
					#~ TRAJ.append((frame,time,area, cellXY, celllabels,fl0, fl1))
				#~ else:
					#~ TRAJ.append((frame,time,area, cellXY, celllabels,fl0))
				


				#~ #pdb.set_trace()
				#~ TRAJCOUNT += 1
				#~ print TRAJCOUNT, ' trajectories'
				

	#~ print TRAJCOUNT, ' total trajectories'
	
	#~ #pickle file to output when running tests
	#~ with open('raw_traj.pkl', 'wb') as f:
		#~ pickle.dump(TRAJ, f, protocol=pickle.HIGHEST_PROTOCOL)
	# pdb.set_trace()

	with open('raw_traj.pkl', 'rb') as f:
		TRAJ = pickle.load(f)

	########################################################################
	########################################################################
					
	# keep track of which indices are visited
	LVISITED = []
	#keep track of previous locations
	LOC = {}
	#keep track of labels in each frame
	LABELS= {}
	#keep dictionary of images to save at the end
	images = {}			
	#store trajectories with new names				
	NEWTRAJ = {}
	#keep track of which trajectories are branches
	CHANGEDTRAJ = {}
	#keep track of how many branches a given trajectory has
	BRANCH = {}
	#keep track of all cell labels
	TOWRITE = {}
	#keep track of cell divisions
	DIVIDE = {}


	colors = getColors()
	trajnum = -1

	#actually go through all the trajectories
	for traj in TRAJ:
		frame,time,area, cellXY, celllabels,fl0 = traj
		trajnum= trajnum+1

		print 'traj = ', trajnum, 'cellXY = ', len(cellXY)
		
		#eachtraj represents each frame a trajectory is in
		for eachtraj in range(0,len(cellXY)):
			cell = cellXY[eachtraj]
			label = celllabels[eachtraj]
			
			if len(label) > 900000:
				print 'skipping traj ', trajnum, 'label is too large'
				
			else:	
				#print cell
				iFRAME = frame[eachtraj]
				#reads all the frame images into images dictionary 
				if iFRAME not in LVISITED:
					LVISITED.append(iFRAME)
					filename = rootdir + labelimage % (iFRAME)
					img = ndimage.imread(filename)
					imgshape = img.shape
					bimg = np.zeros(imgshape  + (3,))
					img = addGrayToColor(bimg, img)
					images[iFRAME] = img
					LOC[iFRAME] = {}
					LABELS[iFRAME] = {}
					TOWRITE[iFRAME] = []
				#area = flarea[eachtraj]
				getPositions(trajnum, cell, iFRAME, label)


	########################################################################
	########################################################################

	LVISITED = sorted(LVISITED)
	for iFRAME in LVISITED:
		relableFRAME(iFRAME)

	NEWTRAJLIST = []
			
	#make NEWTRAJ list using new TRAJ names
	for key in NEWTRAJ:
		traj = TRAJ[key]
		frame,time,area,cellXY,celllabels,fl0 = traj
		trajname = NEWTRAJ[key]
		#calculate doubling times
		try:
			divisions = DIVIDE[key]
		except:
			divisions = None
		try:
			j=0
			k=1
			dframes = []
			for divide in range(len(divisions)-1):
				dframe = divisions[k] - divisions[j]
				dframes.append(dframe)
				j +=1
				k +=1
			dframes = np.array(dframes)
			dtime = (np.mean(dframes))*dt
			if dtime == 0:
				dtime = 'nan'
		except:
			dtime = 'nan'
		traj = trajname,frame,time,area,cellXY,celllabels,fl0,divisions,dtime
		NEWTRAJLIST.append(traj)
		print len(NEWTRAJLIST), 'lineages'
		
	if writeLabels:
		if writeNumbers:
			#pdb.set_trace()
			#write text to images and save
			for key in images:
				image = writeText(key)
				print 'saving ... ', key
				savename = rootdir + labelname %(key)
				misc.imsave(savename, image)

		else:
			for key in images:
				image = images[key]
				print 'saving ... ', key
				savename = rootdir + labelname %(key)
				misc.imsave(savename, image)

	########################################################################
	########################################################################

	#output CSV

	#things to add to csvfile
	length = []
	name = []
	itimes = []
	etimes = []
	meanAREA = []
	stdAREA = []
	meanFL = []
	stdFL = []
	dtimes = []


	for traj in NEWTRAJLIST:
		trajname,frame,time,area,cellXY,celllabels,fl0,divisions,dtime = traj
		name.append(trajname)
		itimes.append(time[1])
		etimes.append(time[-1])
		area= np.array(area)
		meanAREA.append(np.mean(area))
		stdAREA.append(np.std(area))
		fl0 = np.array(fl0, dtype = np.float)
		meanFL.append(np.nanmean(fl0))
		stdFL.append(np.nanstd(fl0))
		dtimes.append(dtime)




	NAMES = np.array(name)
	ITIMES = np.array(itimes)
	ETIMES = np.array(etimes)
	MEANAREA = np.array(meanAREA)
	STDAREA = np.array(stdAREA)
	MEANFL = np.array(meanFL)
	STDFL = np.array(stdFL)
	DTIMES = np.array(dtimes)

	f = open(lineageCSV, 'wb')
	f.write('traj name,')
	f.write('final time,')
	f.write('initial time,')
	f.write('mean area,')
	f.write('std. area,')
	f.write('mean fluorescence,')
	f.write('std. fluorescence,')
	f.write('doubling time')
	f.write('\n')

	for ijk in range(len(name)):
		f.write(str(NAMES[ijk]) + ',')
		f.write(str(ITIMES[ijk]) + ',')
		f.write(str(ETIMES[ijk]) + ',')
		f.write(str(MEANAREA[ijk]) + ',')
		f.write(str(STDAREA[ijk]) + ',')
		f.write(str(MEANFL[ijk]) + ',')
		f.write(str(STDFL[ijk]) + ',')
		f.write(str(DTIMES[ijk]) + ',')
		f.write('\n')			

	f.close()	




	#save picklefile with NEWTRAJLIST
	#traj = frame,time,area, cellXY, celllabels,fl0
	with open('lineagetracking.pkl', 'wb') as f:
		pickle.dump(NEWTRAJLIST, f, protocol=pickle.HIGHEST_PROTOCOL)

	with open('lineagetrackingsummary.pkl', 'wb') as f:
		pickle.dump((NAMES,ITIMES,ETIMES,MEANAREA,STDAREA,MEANFL,STDFL), f, protocol=pickle.HIGHEST_PROTOCOL)

	#saves TOWRITE in pkl file to use in Image_Analysis.ipynb
	with open('lineagetext.pkl', 'wb') as f:
		pickle.dump(TOWRITE, f, protocol=pickle.HIGHEST_PROTOCOL)


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
		#Directory containing the images
		ImageDir = 'Aligned'
		AlignDir = ImageDir
		#get working directory
		WorkDir = os.getcwd()
		#Image filename preceding channel indication (e.g. 20171212_book)
		fname = '20171212_book'
		
		#mask directory (relative to image directory) 
		Mask2Dir = 'Mask2'
		
		#first frame of images
		FIRSTFRAME = 1
		#last frame of images
		FRAMEMAX = 467
		#first frame with fluorescence image
		FLINITIAL = 9
		#frequency of fluorescence images (i.e. every nth frame)
		FLSKIP = 10
		#time between frames in minutes
		Ftime = 0.5 #min
		#number of fluorescence channels
		FLChannels = 1

		#labels for fluorescence channels (must be strings)
		FLLABELS = ['GFP']
		
		#name of directory to save lineages into relative to image directory
		LineageDir = 'Lineages'
		if not os.path.isfile(AlignDir + '/' + LineageDir):
			os.system('mkdir ' + AlignDir + '/' + LineageDir)
		#minimum area of cell in pixels
		AREAMIN = 100
		#maximum area of cell in pixels
		AREAMAX = 2500
		#minimum length of trajectories in # of frames
		MINTRAJLENGTH = 25
		
		
		TrackARG = [AlignDir, fname, Mask2Dir, LineageDir, FIRSTFRAME, FRAMEMAX, AREAMIN, AREAMAX,FLLABELS, Ftime, FLSKIP, FLINITIAL,MINTRAJLENGTH]
		main(TrackARG)
