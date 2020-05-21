# CellTracking
Python and ImageJ scripts designed to track single cells, determine cell lineage and analyze fluorescence

Segmentation Tools for Bacteria based on Machine Learning. Version 3.1. June 2019. The purpose of this project is to provide a number of tools for cell-tracking and image analysis. These tools consist of a number of scripts that utilize preexisting open-source software (Fiji ImageJ, Anaconda 3.7, and related packages) included here and available online (https://www.anaconda.com/download/, https://imagej.net/Fiji/Downloads). 

-----------------------------------------------------------------------------------------------------------------------------
Publication

Methodology for this repository was first published in Deter et al. 2019. A Cell Segmentation/Tracking Tool Based on Machine Learning. Methods Mol Bio. (https://doi.org/10.1007/978-1-4939-9686-5_19), and a video tutorial corresponding to version 2.0 is available on Youtube (https://www.youtube.com/watch?v=wGdIvzBevLM&list=PLL9QX_pyUva9e-Nr0xphPegYg2RJj8OQE). For more information and examples in the context of bacterial growth see https://osf.io/75avy/.

-----------------------------------------------------------------------------------------------------------------------------
Versions

For version 2.0 see https://github.com/hdeter/CellTracking/releases/tag/v2.0.

Noted updates for version 3.1:
1. Filenames are expected to be in the following format name, 6 digit number, xy, 1 digit number, c, 1 digit number; e.g. name000001xy1c1.tif
2. The pipeline can now process up to 9 xy regions simultaneously
3. The pipeline can now process up to 9 fluorescence channels simultaneously
4. Videos with moving graphs now show single cell data when cells are numbered in image video.
5. The pipeline can now stack moving graphs and image videos into a single video.
6. Prompts can now be answered in a csv file rather than in the terminal.


-----------------------------------------------------------------------------------------------------------------------------
Software Installation in Ubuntu 16.04 LTS (Linux)

1. Install Anaconda 3.7 sh file using the following commands; replace /path/to/script.sh with 
the path to the file on your computer:
	sudo chmod +x /path/to/script.sh
	/path/to/script.sh
2. Add conda to the PATH: 
	export PATH=~/anaconda3/bin:$PATH
3. Install OpenCV through Anaconda: 
	conda install opencv
4. Install avconv: 
	sudo apt install libav-tools
5. Download and install Fiji ImageJ (available here: https://imagej.net/Fiji/Downloads)

-----------------------------------------------------------------------------------------------------------------------------
Software Installation in Mac (OS X)

1. Download and install Anaconda 3.7 (https://www.anaconda.com/download/#macos)
2. Install OpenCV through Anaconda: 
conda install opencv
3. Install avconv: 
brew install libav
4. Download and install Fiji ImageJ (https://imagej.net/Fiji/Downloads)

-----------------------------------------------------------------------------------------------------------------------------
Description of scripts (all Python scripts are Python 3.7)

SegmentandTrack.py: Runs the entire cell segmentation and tracking pipeline (or parts of the pipeline) based on user responses to command prompts. The scripts that are called by SegmentandTrack.py are described below.

**Image Alignment (optional)**

Image_alignment.py: Called by SegmentandTrack.py. Uses Fast Fourier Transform (FFT) to calculate alignment for a series of images, and outputs aligned images. Designed to be used for phase and fluorescence images. To improve alignment, a region of interest can be input for use during FFT calculations. See ROI_Align.csv for a sample ROI file.

**Fiji ImageJ scripts**

RunWeka.py: Called by SegmentandTrack.py. For use with Mac, requires an instance if Fiji ImageJ to be open when training a classifier. Calls Segmentation.ijm when training a classifier in the Fiji Weka Segmentation tool, and calls Batch_segmentation.bsh when classifying a group of images with an existing classifier. 

Segmentation.ijm: An ImageJ macro containing a number of user prompts to train a classifier in the Weka Segmentation Tool. Must be run in ImageJ.

Batch_segmentation.bsh: A BeanShell script written for use with Fiji that must be run in the terminal. It uses an existing classifer to train a batch The command to run the script is in RunWeka.py.

**Cell and lineage tracking**

TrackCellLineages.py: Called by SegmentandTrack.py. Uses binary (black and white) masks to identify single cells and obtain quantitative data (area, location, fluorescence, etc.) that is output in a csv and a pickle file. Tracks cells across multiple frames and determines cell lineage. Lineage data is output in a csv and a pickle file. 

**Video rendering and whole image analysis**

Image_analysis_stack.py: Called by SegmentandTrack.py.  Analyzes and renders videos for a group of phase and fluorescence images for global (whole image) fluorescence and fluorescence within a region of interest based on a csv file (optional). This data is output in a csv file. Also outputs several videos that can include cells outlined based on masked, cells numbered based on tracking, and moving graphs that show fluorescence over time. The graphs and images can be stacked together if number of fluorscence channels is less than 4. To adjust image sizes in stacked videos change FACTOR and WPIXEL.


-----------------------------------------------------------------------------------------------------------------------------
Practice dataset

A sample dataset in the context of bacterial growth is available to test the scripts. See https://osf.io/sq49f/.

-----------------------------------------------------------------------------------------------------------------------------
Contact information:

Heather S. Deter
Graduate Research Assistant
South Dakota State University
Email: hdeter2013@gmail.com

Dr. Nicholas C. Butzin
Assistant Professor of Synthetic Biology
South Dakota State University
Email: nicholas.butzin@gmail.com
