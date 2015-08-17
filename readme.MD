This repository contains the model described in the Cognitive Science paper "Concepts as Semantic Pointers: A Framework and Computational Model" (in press). Below are instructions to get the model running and to replicate the results presented in the paper.



Installation
============

Running the model requires installing a development version of the Nengo library for simulating large-scale neural systems. This version is included in the repository. To install this version of Nengo after you've satisfied all of the installation requirements listed below, please do the following:

```
git clone https://github.com/pblouw/blouw-etal-2015.git
cd nengo
python setup.py develop --user
```

If you have a different version of nengo already installed on your machine, you should install into a virtualenv or an Anaconda environment, in which case you can omit the '--user' flag. 

To make raster plots of the sort presented in the paper, you'll need to also install the included package nengo_plot in the same way:

```
cd nengo_plot
python setup.py develop --user

```

Again, the '--user' flag can be omitted if you are using virtualenv or something equivalent. 

Installation requirements
-------------------------

Nengo requires numpy, a library for scientific computing in python. Nengo_plot additionally requires scipy and matplotlib. 
The easist way to get these packages is to install [Anaconda](https://store.continuum.io/cshop/anaconda/), a free python distribution that includes everything you need. Simply install Anaconda and then ensure that it is your default python installation (or, create an environment with each dependency and activate this environment from the command line before you proceed)

Running experiments
-------------------

To run a particular experiment, modify run.py to execute the function associated with a particular study (Posner & Keele, Regehr & Brooks, or Lin & Murphy). You can also modify the function definitions to alter the distortion value (i.e. sigma) applied to the stimuli in a particular experiment. 

Once a simulation is complete, the results will be written to file in the folder 'results/'. You can then regenerate the plots in the paper by running the script 'results.py' in this folder. To generate raster plots of the kind shown in the paper, run a particular experiment in the manner just described, but pass the parameter 'raster=True' to function that runs the experiment. This will write a rasterplot to file after each trial in the experiment. By default, rasterplots are not generated. 

Further documentation is included in each of the scripts contained in this repository, but if you have any questions after reading through this material, please contact me at pblouw@uwaterloo.ca. Note that a small post-publication correction has been added to the pdf of supplementary material that describes the method for obtaining rule weights for the simulations involving the Lin & Murphy experiments. This correction is also documented in the file weights.py


