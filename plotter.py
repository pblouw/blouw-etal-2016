import nengo.spa as spa
import matplotlib.pyplot as plt
import os
import numpy as np
import nengo_plot
from nengo.utils.matplotlib import rasterplot


class Plotter(object):
    """
    Creates experiment-specific plots from an instance of the concept model
    and saves these to file. 
    
    Parameters:
    -----------
    sim : nengo.Simulator object
    	Contains the data from a simulated run of the concept model.
    model : ConceptModel object
        An instance of the concept model with attributes containing probe 
        data from a model run. These attributes are used to generate the 
        plots for each module of the model.
    """
    def __init__(self, model, sim):
        self.sim = sim
        self.task = model.stimuli.task

        # Representation data
        self.vision = model.pVision
        self.context = model.pContext
        self.memory = model.pMemory
        self.motor = model.pMotor
        self.probe = model.probe
        self.actions = model.pActions
        self.apps = model.pApps 
        self.score = model.pScore
        self.decision = model.pDecision
        self.utility = model.pUtility

        # Spike Raster data
        self.visSpikes = model.visSpikes
        self.conSpikes = model.conSpikes
        self.memSpikes = model.memSpikes
        self.motSpikes = model.motSpikes
        self.scoSpikes = model.scoSpikes
        self.appSpikes = model.appSpikes

    def plot(self):
        plots = 6
        fig = plt.figure(figsize=(20,10))

        p1 = fig.add_subplot(plots,1,1)
        p1.plot(self.sim.trange(),spa.similarity(self.sim.data,self.vision))
        p1.set_title('Visual Buffer', fontsize='15')

        p2 = fig.add_subplot(plots,1,2)
        p2.plot(self.sim.trange(),spa.similarity(self.sim.data,self.context))
        p2.set_title('Task Context', fontsize='15')

        p3 = fig.add_subplot(plots,1,3)
        p3.plot(self.sim.trange(),self.sim.data[self.memory])
        p3.set_title('Memory', fontsize='15')

        p4 = fig.add_subplot(plots,1,4)
        p4.plot(self.sim.trange(), self.sim.data[self.score])
        p4.set_title('Coherence Score', fontsize='15')

        p5 = fig.add_subplot(plots,1,5)
        p5.plot(self.sim.trange(),spa.similarity(self.sim.data,self.decision))
        p5.set_title('Decision Buffer', fontsize='15')
        p5.set_ylim([-0.5,2])

        p6 = fig.add_subplot(plots,1,6)
        p6.plot(self.sim.trange(), spa.similarity(self.sim.data, self.motor))
        p6.set_title('Motor', fontsize='15')
        p6.legend(self.motor.target.vocab.keys, fontsize=12, 
            loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=14)

        fig.subplots_adjust(hspace=0.65)
        fig.savefig(os.path.join('results', str(self.probe)))
        plt.close(fig)

    def plot_spikes_vis(self):
        plot = nengo_plot.Time(self.sim.trange())
        plot.add('Visual Buffer', spa.similarity(self.sim.data, self.vision), 
                 overlays=[(0.06,'Posner Task'),(0.25, 'Test Stimulus')])
        plot.add_spikes('', self.sim.data[self.visSpikes], 
                        sample_by_variance=64, cluster=True)

        plot.add('Memory - Context', spa.similarity(self.sim.data, self.context),
                 overlays=[(0.085,'Perceptual Evaluation'),
                           (0.14,'Motor Routing')])
        plot.add_spikes('', self.sim.data[self.conSpikes], 
                        sample_by_variance=64, cluster=True)

        plot.add('Memory - SP', spa.similarity(self.sim.data, self.memory),
                 overlays=[(0.25,'Semantic Pointer')])
        plot.add_spikes('', self.sim.data[self.memSpikes],
                        sample_by_variance=64, cluster=True)

        plot.add('Motor Buffer', spa.similarity(self.sim.data, self.motor),
                 overlays=[(0.25,'Category Label A')])
        plot.add_spikes('', self.sim.data[self.motSpikes], 
                        sample_by_variance=64, cluster=True)
        plot.save('vis_raster.png')

    def plot_spikes_inf(self):
        plot = nengo_plot.Time(self.sim.trange())
        plot.add('Visual Buffer', spa.similarity(self.sim.data, self.vision), 
                 overlays=[(0.06,'Lin Task'),(0.25, 'Test Stimulus')])
        plot.add_spikes('', self.sim.data[self.visSpikes], 
                        sample_by_variance=64, cluster=True)

        plot.add('Memory - Context', spa.similarity(self.sim.data, self.context),
                 overlays=[(0.085,'Inferential Evaluation'),(0.14, 'Rule 1'),
                           (0.19, 'Rule 2'),(0.24, 'Rule 3'), 
                           (0.29, 'Rule 4'),(0.35, 'Motor Routing')])
        plot.add_spikes('', self.sim.data[self.conSpikes], 
                        sample_by_variance=64, cluster=True)

        plot.add('Applications', spa.similarity(self.sim.data, self.apps),
                 overlays=[(0.2, 'Feature 1'),
                           (0.25, 'Feature 2'),(0.3, 'Feature 3'), 
                           (0.35, 'Feature 4')])
        plot.add_spikes('', self.sim.data[self.appSpikes],
                        sample_by_variance=64, cluster=True)

        plot.add('Coherence', self.sim.data[self.score],
                overlays=[(0.35, 'Coherence Value')])
        plot.add_spikes('', self.sim.data[self.scoSpikes], sample_by_variance=64)

        plot.add('Motor Buffer', spa.similarity(self.sim.data, self.motor),
                 overlays=[(0.42,'Positive Judgment')])
        plot.add_spikes('', self.sim.data[self.motSpikes], 
                        sample_by_variance=64, cluster=True)
        plot.save('inf_raster.png')
