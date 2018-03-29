# CellTracking
Python and ImageJ scripts designed to track single cells and analyze fluorescence

Segmentation Tools for Bacteria based on Machine Learning. Version 1.0. March 2018. The purpose of this project is to provide a number of tools for cell-tracking and image analysis. These tools consist of a number of scripts that utilize preexisting open-source software (Fiji ImageJ, Anaconda 2.7, and related packages) included here and available online (https://www.anaconda.com/download/, https://imagej.net/Fiji/Downloads). A detailed method for using these tools will be published in Springer Methods (Deter et al. 2018).

-----------------------------------------------------------------------------------------------------------------------------
Software Installation in Ubuntu 16.04 LTS (Linux)

1. Install Anaconda 2.7 (Anaconda2-5.1.0-Linux-x86_64.sh) file using the following commands; replace /path/to/script.sh with 
the path to the file on your computer:
	sudo chmod +x /path/to/script.sh
	path/to/script.sh
2. Add conda to the PATH: 
	export PATH=~/anaconda2/bin:$PATH
3. Install OpenCV through Anaconda: 
	conda install opencv
4. Install avconv: 
	sudo apt install libav-tools
5. Unzip fiji-linux64-20170530.zip

-----------------------------------------------------------------------------------------------------------------------------
Software Installation in Mac (OS X)

1. Download and install Anaconda 2.7 (https://www.anaconda.com/download/#macos)
2. Install OpenCV through Anaconda: 
conda install opencv
3. Install avconv: 
brew install libav
4. Unzip fiji-linux64-20170530.zip

-----------------------------------------------------------------------------------------------------------------------------
Description of scripts (all Python scripts are Python 2.7)

Image_alignment.py: Requires modification of a few parameters at the beginning of the script (see comments). Uses Fast Fourier Transform (FFT) to calculate alignment for a series of images, and outputs aligned images. Designed to be used for phase and fluorescence images. To improve alignment, a region of interest can be input for use during FFT calculations. See ROI_Align.csv for a sample ROI file.

Image_alignment_prompt.py: A user friendly version of Image_alignment.py that requires user input.

Segmentation.ijm: An ImageJ macro containing a number of user prompts to train a classifier in the Weka Segmentation Tool.

Batch_segmentation.bsh: A BeanShell script written for use with Fiji that must be run in the terminal. It uses an existing classifer to train a batch The command to run the script is in RunWeka.py. 

RunWeka.py: Requires modification of a few parameters at the beginning of the script (see comments). For use with Linux. Calls Segmentation.ijm when training a classifier in the Fiji Weka Segmentation tool, and calls Batch_segmentation.bsh 
when classifying a group of images with an existing classifier. 

RunWeka_prompt.py: For use with Linux. A user friendly version of RunWeka.py that requires user input.

RunWekaMac.py: Requires modification of a few parameters at the beginning of the script (see comments). For use with Mac, requires an instance if Fiji ImageJ to be open when training a classifier. Calls Segmentation.ijm when training a classifier in the Fiji Weka Segmentation tool, and calls Batch_segmentation.bsh when classifying a group of images with an existing classifier. 

RunWekaMac_prompt.py: For use with Mac. A user friendly version of RunWekaMac.py that requires user input.

Track-cell-lineages.py: Requires modification of a few parameters at the beginning of the script (see comments). Uses binary (black and white) masks to identify single cells and obtain quantitative data (area, location, fluorescence, etc.) that is output in a csv and a pickle file. Tracks cells across multiple frames and determines cell lineage. Lineage data is output in a csv and a pickle file. 

Track-cell-lineages_prompt.py: A user friendly version of Track-cell-lineages.py that requires user input.

Image_analysis.py: Requires modification of a few parameters at the beginning of the script (see comments). Analyzes and renders videos for a group of phase and fluorescence images for global (whole image) fluorescence and fluorescence within a region of interest based on a csv file (optional). This data is output in a csv file.

Image_analysis_prompt.py: A user friendly version of Image_analysis.py that requires user input.

-----------------------------------------------------------------------------------------------------------------------------
Practice dataset

A sample dataset in the context of bacterial growth is available to test the scripts. The complete image dataset (467 frames) is available on FigShare.

-----------------------------------------------------------------------------------------------------------------------------
Contact information:

Heather S. Deter
Graduate Assistant
South Dakota State University
Email: hdeter2013@gmail.com

Dr. Nicholas C. Butzin
Assistant Professor of Synthetic Biology
South Dakota State University
Email: nicholas.butzin@gmail.com
