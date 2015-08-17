import stimgen
import weights
import numpy as np
from model import ConceptModel

class ExperimentManager(object):
    """
    Runs the model for a specific number of times while logging results and 
    computing performance accuracy on a particular experimental task. For 
    example, the Experiment Manager will run the Posner experiment for 32 
    simulated participants with stimuli generated at a particular distortion
    level.

    Parameters:
    -----------
    dimensions : int
        The dimensionality of the representations used in the experiment.
    seeds : list
        A list of random number seeds corresponding to a set of simuluated
        participants in the experiment. The seeds are used to generate the 
        model and fix the selection of neuron parameters, tuning curves, etc.
    dvals : list
        The standard deviation of the gaussian distribution used to distort 
        the stimuli in various experimental conditions. 
    """
    def __init__(self, dimensions, seeds, dvals, raster=False):
        self.dim = dimensions
        self.dvals = dvals
        self.seeds = seeds
        self.raster = raster

    def posner(self):
        tally = np.zeros(4)
        total = np.zeros((len(self.seeds), 4))
        counts = np.array([6,3,6,6], float)
        results = np.zeros((len(self.dvals), 4))
        
        for dval in self.dvals:
            for seed in self.seeds:
                np.random.seed(seed)
                stimuli = stimgen.Posner(self.dim, dval)    
                for stimulus in stimuli.test_vectors:    
                    print 'Testing ', stimulus
                    model = ConceptModel.posner(self.dim, 
                                                stimulus, 
                                                stimuli, 
                                                seed,
                                                raster=self.raster)
                    model.run()
                    if model.result == True:
                        label = self.posner_labels(stimulus) 
                        tally[label] += 1
                    print 'Status: ', np.where(self.seeds == seed)[0][0], \
                           stimulus, tally, seed

                error = 1 - np.divide(tally, counts)
                total[np.where(self.seeds == seed)[0][0], :] += error
                tally = np.zeros(4)

            average = np.divide(np.sum(total, axis=0), float(len(self.seeds)))
            results[np.where(self.dvals == dval)[0][0], :] = average
            total = np.zeros((len(self.seeds), 4))
        
        np.save('results/Posner'+str(self.dvals[0]), results)
        print results
        return results

    def brooks(self):
        total = np.zeros((len(self.seeds), 3))
        counts = np.array([8, 4, 4], float)
        tally = np.zeros(3)
        results = np.zeros((len(self.dvals), 3))
       
        for dval in self.dvals:
            for seed in self.seeds:
                np.random.seed(seed)
                stimuli = stimgen.Brooks(self.dim, dval)
                if np.where(self.seeds == seed)[0][0] < 4:
                    stimuli.rule1()
                elif np.where(self.seeds == seed)[0][0] < 8:
                    stimuli.rule2()
                elif np.where(self.seeds == seed)[0][0] < 12:
                    stimuli.rule3()
                else:
                    stimuli.rule4()
                for stimulus in stimuli.test_vectors:
                    print 'Testing ', stimulus
                    model = ConceptModel.brooks(self.dim, 
                                                stimulus, 
                                                stimuli, 
                                                seed,
                                                raster=self.raster)
                    model.run()

                    if model.result == True:
                        label = self.brooks_labels(stimulus, stimuli) 
                        tally[label] += 1
                    print 'Status: ', np.where(self.seeds == seed)[0][0], \
                           stimulus, tally, seed

                error = 1 - np.divide(tally, counts)
                total[np.where(self.seeds == seed)[0][0], :] = error
                tally = np.zeros(3)

            average = np.divide(np.sum(total, axis=0), float(len(self.seeds)))
            results[np.where(self.dvals == dval)[0][0], :] = average
            total = np.zeros((len(self.seeds), 3))

        np.save('results/Brooks'+str(self.dvals[0]), results)
        print results
        return results

    def murphy(self):
        tally = np.zeros(4)
        counts = np.array([8, 8, 8, 8], float)
        results = np.zeros((len(self.seeds), 4))

        for seed in self.seeds:
            stimuli = stimgen.Murphy(self.dim, self.dvals)
            weight = weights.murphy(seed)     
            for stimulus in stimuli.stimuli:
                print 'Testing ', stimulus
                c = self.count(stimulus)
                w = weight[0][c,:]
                p = weight[1][c,:]   
                print 'Weights', w
                print 'Products', p
                if 'Prototype' in stimulus:
                    s = [1,1,1,1]
                    model = ConceptModel.murphy(self.dim, 
                                                s, stimuli, 
                                                w, c, seed, 
                                                raster=self.raster)
                    model.run()
                elif 'ConsistentA' in stimulus:
                    s = [1,0,1,1]
                    model = ConceptModel.murphy(self.dim, 
                                                s, stimuli, 
                                                w, c, seed,
                                                raster=self.raster)
                    model.run()
                elif 'ConsistentB' in stimulus:
                    s = [0,1,1,1]
                    model = ConceptModel.murphy(self.dim, 
                                                s, stimuli, 
                                                w, c, seed,
                                                raster=self.raster)
                    model.run()
                else:
                    s = [0,0,0,1]
                    model = ConceptModel.murphy(self.dim, 
                                                s, stimuli, 
                                                w, c, seed,
                                                raster=self.raster)
                    model.run()
                if model.result == True:
                    label = self.murphy_labels(stimulus)
                    tally[label] += 1
                print 'Status: ', np.where(self.seeds == seed)[0][0], \
                       stimulus, tally, seed
                
            error = np.divide(tally, counts)
            results[np.where(self.seeds == seed)[0][0], :] = error
            tally = np.zeros(4)

        average = np.divide(np.sum(results, axis=0), float(len(self.seeds)))
        np.save('results/Murphy', average)

        print average
        return average

    def posner_labels(self, stimulus):
        if 'T' in stimulus:
            return 0
        elif 'L' in stimulus:
            return 2
        elif 'H' in stimulus:
            return 3
        else:
            return 1

    def brooks_labels(self, stimulus, stimuli):
        if stimuli.labelled_stimuli[stimulus][1] == 'Training Item':
            return 0
        elif stimuli.labelled_stimuli[stimulus][1] == 'Good Transfer':
            return 1
        else:
            return 2

    def murphy_labels(self, stimulus):
        if 'Prototype' in stimulus:
            return 0
        elif 'ConsistentA' in stimulus:
            return 1
        elif 'ConsistentB' in stimulus:
            return 2
        else:
            return 3

    def count(self, stimulus):
        if '0' in stimulus:
            count = 0
        elif '1' in stimulus:
            count = 1
        elif '2' in stimulus:
            count = 2
        elif '3' in stimulus:
            count = 3
        elif '4' in stimulus:
            count = 4
        elif '5' in stimulus:
            count = 5
        elif '6' in stimulus:
            count = 6
        else:
            count = 7
        return count