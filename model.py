import numpy as np
import nengo
import nengo.spa as spa
from nengo.networks import Product, CircularConvolution
from plotter import Plotter

class ConceptModel(object):
    """
    Class that describes the model used in all experiments. Allows
    a simulation to run for a specific amount of time, and also allows 
    the model's vocabularies and stimuli to initialized correctly for
    a specific experimental task.

    Parameters:
    -----------
    probe : Semantic Pointer label
        The label of the semantic pointer to be categorized during the
        simulation (e.g. an SP corresponding to a particular dot pattern 
        or drawing)
    stimuli : stimgen.Stimuli object
         Contains all the stimuli vectors used in a given experimental
         task. These vectors are added to the SP vocabularies described 
         below when an instance of the model is generated.
    main_voc : spa.Vocabulary object
        A vocabularly containing all SP's used in a given experiment. 
    feat_voc : spa.Vocabulary object
        A vocabulary containing SP's for all the features that inferential
        rules are defined over (defaults to label_voc in any non-inferential
        task)
    weight_voc : spa.Vocabulary object
        A vocabulary containing SP's for the *weighted* features that 
        inferential rules produce (defaults to label_voc in any
        non-inferential task)
    label_voc : spa.Vocabulary object
        A vocabulary containing SP's for all the labels that can be used to 
        categorize a probe stimulus.    
    seed : int
        The random number seed used to generate all neuron parameters in 
        the model.
    """
    def __init__(self, probe, stimuli, main_voc, feat_voc, 
                 weight_voc, label_voc, seed, raster):
        self.dimensions = stimuli.dimensions
        self.probe = probe
        self.stimuli = stimuli
        self.main_voc = main_voc
        self.feat_voc = feat_voc
        self.weight_voc = weight_voc
        self.label_voc = label_voc
        self.seed = seed
        self.raster = raster

    def run(self):
        n_neurons = 40
        subdim = 16
        direct = False
        output = self.stimuli.output # vector output from coherence score
        threshold = self.stimuli.threshold # for correct answer on inf task
        g_transform = np.transpose(np.ones((1, self.dimensions))) # mem gate
        
        model = spa.SPA(label='ConceptModel', seed=self.seed)
        
        with model:    

            # Vision Component
            model.vision = spa.Buffer(
                dimensions=self.dimensions, subdimensions=subdim, 
                neurons_per_dimension=n_neurons, vocab=self.main_voc, 
                direct=direct) 

            # Memory Component 
            model.sp_mem = spa.Buffer(
                dimensions=self.dimensions, subdimensions=subdim,
                neurons_per_dimension=n_neurons, vocab=self.main_voc, 
                direct=direct)
            model.context_mem = spa.Memory(
                dimensions=self.dimensions, subdimensions=subdim,
                neurons_per_dimension=n_neurons, vocab=self.main_voc,
                tau=0.01/0.3, direct=direct)

            # Inferential Evaluation Subsystem
            model.inference = spa.Buffer(
                dimensions=self.dimensions, subdimensions=subdim, 
                neurons_per_dimension=n_neurons, vocab=self.feat_voc, 
                direct=direct)
            model.score = spa.Memory(
                dimensions=1, neurons_per_dimension=3000, 
                synapse=0.05, direct=direct)
            model.gatesignal = spa.Memory(
                dimensions=1, neurons_per_dimension=500,
                synapse=0.05, direct=direct)         
            model.cleanup1 = spa.AssociativeMemory(
                input_vocab=self.feat_voc, output_vocab=self.weight_voc, 
                n_neurons_per_ensemble=250, threshold=0.25)
 
            # Perceptual Evaluation Subsystem
            model.cleanup2 = spa.AssociativeMemory(
                input_vocab=self.label_voc, output_vocab=self.label_voc, 
                n_neurons_per_ensemble=1000, threshold=threshold)   

            # Shared Component
            model.decision = spa.Memory(
                dimensions=self.dimensions, subdimensions=subdim,
                neurons_per_dimension=n_neurons, vocab=self.label_voc,
                tau=0.01/0.3, synapse=0.05, direct=direct)
            
            # Motor Component
            model.motor = spa.Memory(
                dimensions=self.dimensions, subdimensions=subdim,
                neurons_per_dimension=n_neurons, vocab=self.label_voc,
                synapse=0.1, direct=direct)

            # Convenient probing of rule applications
            if self.raster:
                model.apps = spa.Buffer(
                    dimensions=self.dimensions, subdimensions=subdim, 
                    neurons_per_dimension=n_neurons, vocab=self.feat_voc, 
                    direct=direct)

                self.pApps = nengo.Probe(
                    model.apps.state.output, synapse=0.03)
                self.appSpikes = nengo.Probe(
                    model.apps.state.ea_ensembles[3], 'spikes')
                
                nengo.Connection(model.cleanup1.output, model.apps.state.input)
            
            # Action definitions
            actions = spa.Actions(
                'dot(vision, POSNER) --> context_mem=VIS',
                'dot(vision, BROOKS) --> context_mem=VIS',
                'dot(vision, MURPHY) --> context_mem=INFER',
                'dot(context_mem, VIS)   --> gatesignal=2, context_mem=R5',
                'dot(context_mem, INFER) --> context_mem=R1',
                'dot(context_mem, R1)    --> inference=sp_mem*~R1, context_mem=R2',
                'dot(context_mem, R2)    --> inference=sp_mem*~R2, context_mem=R3',
                'dot(context_mem, R3)    --> inference=sp_mem*~R3, context_mem=R4',
                'dot(context_mem, R4)    --> inference=sp_mem*~R4, context_mem=R5',
                'dot(context_mem, R5)    --> motor=decision'
                )

            # Basal ganglia and thalamus
            model.bg = spa.BasalGanglia(actions)
            model.thal = spa.Thalamus(model.bg)  

            # Subnetworks defined outside of SPA
            model.product = Product(600, self.dimensions) 
            model.vis_gate = Product(300, self.dimensions)
            model.mem_gate = Product(300, self.dimensions)
            model.conv = CircularConvolution(250, self.dimensions,invert_a=True)

            # Connections for gate with memory for perceptual evaluation tasks
            nengo.Connection(model.vision.state.output, model.vis_gate.B)
            nengo.Connection(model.gatesignal.state.output, model.vis_gate.A,
                transform=g_transform)
            nengo.Connection(model.vis_gate.output, model.conv.A)
            nengo.Connection(model.sp_mem.state.output, model.mem_gate.B)
            nengo.Connection(model.gatesignal.state.output, model.mem_gate.A,
                transform=g_transform)
            nengo.Connection(model.mem_gate.output, model.conv.B)
            nengo.Connection(model.conv.output, model.decision.state.input)

            # Connections for inferential evaluation tasks
            nengo.Connection(model.inference.state.output, model.cleanup1.input)
            nengo.Connection(model.cleanup1.output, model.product.A)
            nengo.Connection(model.vision.state.output, model.product.B)
            nengo.Connection(model.cleanup2.output, model.decision.state.input)
            nengo.Connection(model.product.output, model.score.state.input, 
                transform=model.product.dot_product_transform())
            nengo.Connection(model.score.state.output, model.cleanup2.input,
                transform=np.transpose(output))
            
            # Input to visual buffer
            def vision(t):
                if t<0.08: return self.stimuli.task
                if 0.0801 <= t < 1: return self.probe
                else: return '0'

            model.input = spa.Input(vision=vision, sp_mem=self.stimuli.memory) 

            # Inhibit the gate with inference actions
            actions = [2,4,5,6,7,8]
            target = model.gatesignal
            for action in actions:
                for e in target.all_ensembles:
                    nengo.Connection(model.thal.actions.ensembles[action],
                                     e.neurons, transform=[[-2]]*e.n_neurons)

            # Set radius for ensemble computing scalar coherence score.
            for ens in model.score.state.ensembles:
                ens.radius = 2 

            # Define probes for semantic pointer plots
            self.pVision = nengo.Probe(
                model.vision.state.output, synapse=0.03)
            self.pMotor = nengo.Probe(
                model.motor.state.output, synapse=0.03)
            self.pMemory = nengo.Probe(
                model.sp_mem.state.output, synapse=0.03)
            self.pContext = nengo.Probe(
                model.context_mem.state.output, synapse=0.03)
            self.pScore = nengo.Probe(
                model.score.state.output, synapse=0.03)
            self.pDecision = nengo.Probe(
                model.decision.state.output, synapse=0.03)
            self.pActions = nengo.Probe(
                model.thal.actions.output, synapse=0.01)
            self.pUtility = nengo.Probe(
                model.bg.input, synapse=0.01)

            # Define probes for spike rasters
            self.visSpikes = nengo.Probe(
                model.vision.state.ea_ensembles[3], 'spikes')
            self.conSpikes = nengo.Probe(
                model.context_mem.state.ea_ensembles[5],'spikes')
            self.memSpikes = nengo.Probe(
                model.sp_mem.state.ea_ensembles[7], 'spikes')
            self.motSpikes = nengo.Probe(
                model.motor.state.ea_ensembles[1], 'spikes')
            self.scoSpikes = nengo.Probe(
                model.score.state.ea_ensembles[0], 'spikes')

        # Run the model
        sim = nengo.Simulator(model)
        sim.run(0.45)

        # Save graph if plotting chosen
        if self.raster:
            graph = Plotter(self, sim)
            if self.stimuli.task == 'MURPHY':
                graph.plot_spikes_inf() 
            else:
                graph.plot_spikes_vis()

        # Assess correctness of output
        self.measure_output(sim)

    def measure_output(self, sim):
        # Determines whether the simulation resulted in a correct decision
        time = 425
        label = self.stimuli.get_label(self.probe)
        value = np.dot(sim.data[self.pMotor][time], 
                       self.label_voc.pointers[label].v)
        
        # Avoid roundoff errors
        top = max(self.label_voc.dot(sim.data[self.pMotor][time])) - 0.005

        if (value >= 0.5) and (value >= top):
            self.result = True
            print value, self.result
        else:
            self.result = False
            print value, self.result

    @classmethod
    def posner(cls, dimensions, probe, stimuli, seed, raster=False):
        # Initializes an instance of the model that performs the Posner task
        main_voc = spa.Vocabulary(dimensions)
        feat_voc = spa.Vocabulary(dimensions)
        weight_voc = spa.Vocabulary(dimensions)
        label_voc = spa.Vocabulary(dimensions)

        stimuli.threshold = 0.5
        
        if probe not in stimuli.train_vectors:
            main_voc.add(probe, stimuli.test_vectors[probe])

        ConceptModel.add_to(main_voc, stimuli.train_vectors)
        ConceptModel.add_to(main_voc, stimuli.label_vectors)
        ConceptModel.add_to(feat_voc, stimuli.label_vectors)
        ConceptModel.add_to(weight_voc, stimuli.label_vectors)
        ConceptModel.add_to(label_voc, stimuli.label_vectors)

        return ConceptModel(probe, stimuli, main_voc, feat_voc,
                            weight_voc, label_voc, seed, raster)

    @classmethod
    def brooks(cls, dimensions, probe, stimuli, seed, raster=False):
        # Initializes an instance of the model that performs the Brooks task
        main_voc = spa.Vocabulary(dimensions)
        feat_voc = spa.Vocabulary(dimensions)
        weight_voc = spa.Vocabulary(dimensions)
        label_voc = spa.Vocabulary(dimensions)

        stimuli.threshold = 0.5

        ConceptModel.add_to(main_voc, stimuli.stimuli_A)
        ConceptModel.add_to(main_voc, stimuli.label_vectors)
        ConceptModel.add_to(feat_voc, stimuli.label_vectors)
        ConceptModel.add_to(weight_voc, stimuli.label_vectors)
        ConceptModel.add_to(label_voc, stimuli.label_vectors)

        if probe not in stimuli.stimuli_A.keys():
            main_voc.add(probe, stimuli.test_vectors[probe])
            
        x = stimuli.labelled_stimuli[probe]
        print 'Correct Labels'
        print x[0], x[1]

        return ConceptModel(probe, stimuli, main_voc, feat_voc, 
                            weight_voc, label_voc, seed, raster)

    @classmethod
    def murphy(cls, dimensions, probe, stimuli, weights, count, seed, raster=False):
        # Initializes an instance of the model that performs the Murphy task
        features = sorted([w for w in stimuli.features.keys() 
                           if str(count) in w])

        probe = '%d*A%d+%d*B%d+%d*C%d+%d*D%d'% \
        (probe[0], count, probe[1], count, probe[2], count, probe[3], count)
        stimuli.memory = 'R1*A%d+R2*B%d+R3*C%d+R4*D%d'% \
        (count, count, count, count)

        main_voc = spa.Vocabulary(dimensions)
        feat_voc = spa.Vocabulary(dimensions)
        weight_voc = spa.Vocabulary(dimensions)
        label_voc = spa.Vocabulary(dimensions)
        
        for label in stimuli.label_vectors:
            main_voc.add(label, stimuli.label_vectors[label])

        for feature in features:
            main_voc.add(feature, stimuli.features[feature])
            feat_voc.add(feature, stimuli.features[feature])
            weight_voc.add(feature, stimuli.features[feature]*
                   1.5*weights[features.index(feature)])

        ConceptModel.add_to(label_voc, stimuli.label_vectors)
        stimuli.threshold = 0.724
        
        return ConceptModel(probe, stimuli, main_voc, feat_voc, 
                            weight_voc, label_voc, seed, raster)

    @staticmethod
    def add_to(vocab, stimuli):
        # Helper for adding items to vocabularies
        for item in stimuli:
            vocab.add(item, stimuli[item])
        return vocab