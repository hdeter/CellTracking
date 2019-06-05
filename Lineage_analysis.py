#uses pickle file from lineageTracking and makes graphs


############################################################################
############################################################################

from time import sleep
from time import clock
import math

import numpy as np
import numpy.random as rnd

import pdb

import string

import sys
import os

import pickle as pickle

from multiprocessing import Pool

import scipy.ndimage as ndimage
import scipy.misc as misc

import scipy.sparse

import cv2 as cv

import csv

############################################################################
############################################################################

if (True):
	import matplotlib as mpl
	import matplotlib.pyplot as plt

	mpl.rcParams['legend.fontsize'] = 12

	# change fonts so that PDF's do not export with outlined fonts (bad for editing)
	mpl.rcParams['pdf.fonttype'] = 42

	#fig = plt.figure(figsize=(12,8))

	plt.ion()
	plt.show()

############################################################################
############################################################################

#######################################################################
#######################################################################

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


# helper function for input
def int_input(DISPLAY):
	iValue = None
	while (iValue is None):
		str = input(DISPLAY)
		try:
			iValue = int(str)
		except:
			iValue = None
	return iValue
	

# helper function for input
def float_input(DISPLAY):
	iValue = None
	while (iValue is None):
		str = input(DISPLAY)
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
			print('cannot find file')
	return FILENAME
	
def getdirname(prompt):
	ISPATH = False
	while not ISPATH:
		PATHNAME = text_input(prompt)
		ISPATH = os.path.isdir(PATHNAME)
		if not ISPATH:
			print('cannot find directory')
	return PATHNAME
##################################################################

def text_list():
	LENGTH = int_input('Enter the number of lineages you wish to analyze: ')
	print('Input the full name for each lineage (e.g. 0001, 0001-2, 10001).')
	LIST = []
	for l in range(LENGTH):
		L = text_input('Lineage ' + str(l+1) + ': ')
		LIST.append(L)
	return LIST
		
#################################################################

def CSVOUT(trajname,frame,time,area,cellXY,celllabels,fl0,divisions,dtime,FLLABELS,iXY):

	#calculate doubling time
	#~ if divisions != None:
		#~ #make empty list length of time
		#~ dtimes = []
		#~ for f in frame:
			#~ if f in divisions:
				#~ dtimes.append('Division')
			#~ else:
				#~ dtimes.append(' ')
	
	i = 0
	for fl in fl0:
		if fl[0] == None:
			fl0[i] = 'nan'
		i +=1
			
	FRAME = np.array(frame)
	TIMES = np.array(time)
	AREA = np.array(area)
	CELLXY = np.array(cellXY)
	CELLX = CELLXY[:,1]
	CELLY = CELLXY[:,0]
	FL0 = np.array(fl0)
	#~ if dtimes != None:
		#~ DTIMES = np.array(dtimes)

	f = open(OUTDIR + '/iXY' + str(iXY) + '-' + trajname + '.csv' , 'w')
	f.write('frame,')
	f.write('time,')
	f.write('area,')
	f.write('x position,')
	f.write('y position,')
	for fllabel in FLLABELS:
		f.write(fllabel + ',')
	#~ if dtimes != None:
		#~ f.write('divisions,')
	f.write('\n')

	for ijk in range(len(FRAME)):
		f.write(str(FRAME[ijk]) + ',')
		f.write(str(TIMES[ijk]) + ',')
		f.write(str(AREA[ijk]) + ',')
		f.write(str(CELLX[ijk]) + ',')
		f.write(str(CELLY[ijk]) + ',')
		for fl0 in FL0[ijk]:
			f.write(str(fl0) + ',')
		#~ if dtimes != None:
			#~ f.write(str(DTIMES[ijk]) + ',')
		f.write('\n')			

	f.close()	
	
	CURSOR_UP_ONE = '\x1b[1A'
	print(('saving ' + trajname + '.csv                                 '))                                    
	sys.stdout.write(CURSOR_UP_ONE) 
############################################################################
############################################################################

####################################################################
####################################################################
def main(argv):
	global OUTDIR, iXY
	
	OUTDIR = argv[0]	
	RUNALL = argv[1]
	#list of trajectories to get
	trajnames = argv[2]
	iXY = argv[3]
	#dt = argv[3]
	FLLABELS = argv[4]
	
	
	############################################################################
	############################################################################
	#load files
	try:
		#~ with open('global-cell-statistics.pkl', 'rb') as f:
			#~ fltimeALL,fln,FLMEANALL,FLMEDIANALL,FLSTDALL,FLMEANALLFILTERED,FLMEANALLBACKGROUND = pickle.load(f)

		with open('iXY' + str(iXY) + '_lineagetracking.pkl', 'rb') as f:
			TRAJ = pickle.load(f)

		#~ with open('lineagetrackingsummary.pkl', 'rb') as f:
			#~ NAMES,ITIMES,ETIMES,MEANAREA,STDAREA,MEANFL,STDFL = pickle.load(f)
	except:
		print('Could not load file: '+'iXY' + str(iXY) + '_lineagetracking.pkl /n Cells must be tracked with TrackCellLineages.py (run through SegmentandTrack.py) before analyzing lineages')
	############################################################################
	############################################################################



	DTRAJ = {}
	for traj in TRAJ:
		trajname,frame,time,area,cellXY,celllabels,fl0,divisions,dtime = traj
		if not RUNALL:
			#print trajname
			DTRAJ[trajname] = frame,time,area,cellXY,celllabels,fl0,divisions,dtime
		else:
			CSVOUT(trajname,frame,time,area,cellXY,celllabels,fl0,divisions,dtime,FLLABELS, iXY)
	#pdb.set_trace()
	if not RUNALL:
		for traj in trajnames:
			try:
				#print traj
				frame,time,area,cellXY,celllabels,fl0,divisions,dtime = DTRAJ[traj]
				#pdb.set_trace()
				CSVOUT(traj,frame,time,area,cellXY,celllabels,fl0,divisions,dtime,FLLABELS, iXY)
			except:
				print(traj + ' cannot be found')

############################################################################

def run(argv):
	iXY = argv[0]
	OUTDIR = argv[1]
	if not os.path.isdir(OUTDIR):
		os.system('mkdir ' + OUTDIR)
	RUNALL = argv[2]
	FLLABELS = argv[3]
	
	#list of trajectories to get
	if RUNALL:
		trajnames = None
	else:
		trajnames = text_list()

	#dt = float_input('What is the time per frame in minutes (e.g. 0.5): ')
	

	main([OUTDIR,RUNALL,trajnames,iXY,FLLABELS])

if __name__ == "__main__":
	run()

############################################################################
