import numpy as np
from manager import ExperimentManager

"""
Each experiment can be run by calling the appropriate procedure defined below. 
By default, a single experiment is run at a particular distortion value (sigma), set by
the numpy array passed into the ExperimentManager constructor. Alternatively, 
pass in the array 'dvals' to run a number of experiments sequentially. 
(this will likely take a long time - it is preferable to run multiple experiments 
in parallel, or use python's multiprocessing library if you want to collect a large
number of results).

Set raster = True to save a raster plots during an experiment. This raster plot is
overwritten during each new trial, so you will need to change the code if you want
to save multiple plots.  
"""

dimensions = 128

def run_posner():
	'''Run the posner experiment'''
	np.random.seed(89)
	seeds = np.random.randint(0, 5000, 32)
	dvals = np.arange(0.05, 0.16, 0.01)
	
	manager = ExperimentManager(dimensions, seeds, np.array([0.1]), raster=False)
	manager.posner()

def run_brooks():
	'''Run the brooks experiment'''
	np.random.seed(89)
	seeds = np.random.randint(0, 5000, 16)
	dvals = np.arange(0.01, 0.16, 0.01)
	
	manager = ExperimentManager(dimensions, seeds, np.array([0.02]), raster=False)
	manager.brooks()

def run_murphy():
	'''Run the murphy experiment'''
	np.random.seed(89)
	seeds = np.random.randint(0, 5000, 20)
	
	manager = ExperimentManager(dimensions, seeds, np.array([0.001]), raster=False)
	manager.murphy()

run_murphy()






