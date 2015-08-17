import random 
import numpy as np

class Stimuli(object):
    """
    A parent class for storing parameters and functions that are used
    universally across all stimuli generation procedures. 
    """
    def norm_of_vector(self, v):
        return np.linalg.norm(v)

    def disturb(self, v, std):
        """Adds mean-zero gaussian noise with specified standard dev to v"""
        return v + np.random.normal(loc=0, scale=std, size=len(v))   
        
    def convolve(self, v1, v2):
        return np.fft.ifft(np.fft.fft(v1) * np.fft.fft(v2)).real

    def deconvolve(self, v1, v2):
        return self.convolve(np.roll(v1[::-1],1), v2)

    def cosine(self, v1, v2):
        n1 = self.norm_of_vector(v1)
        n2 = self.norm_of_vector(v2)
        dot = inner_product(v1,v2)
        return dot / (n1*n2)

    def normalize(self, v):
        return v / self.norm_of_vector(v)


class Posner(Stimuli):
    """
    Describes objects that generate and handle stimuli for the Posner and Keele
    experiment involving prototype-based categorization. This object serves 
    as container for the stimuli used in the experiment.

    Parameters:
    -----------
    dimensions : int
        The dimensionality of the stimuli vectors to be generated.
    dval : float
        The standard deviation of the gaussian distribution sampled from to
        'distort' the prototype vectors when generating different classes of
        stimuli. More specifically, the parameter sets the STD of the
        distribution corresponding to the 'low' distortion stimuli, while a 
        fixed multiplier is used to generate the STD of the distribution 
        corresponding to the 'high' distortion stimuli. 
    seed : int
        The seed for the random number generator. Setting this allows for the
        replication of stimuli across runs of the model.
    """
    def __init__(self, dimensions, dval, seed=None):
        self.task = 'POSNER'
        self.prototype_vectors = dict()
        self.train_vectors = dict()
        self.test_vectors = dict()
        self.label_vectors = dict()
        self.dimensions = dimensions
        self.memory = np.zeros(dimensions)
        self.lib = Library()

        # Set seed for random number generator
        if seed != None:
            np.random.seed(seed)

        # Labels and parameters from experimental paper
        prototypes = ['A','B','C']
        labels = ['LabelA', 'LabelB', 'LabelC']
        num_train = 4
        num_per_level = 2
        r = 1.54 # Ratio of high to low distortion values

        # Generate first random prototype vector
        self.lib.add('A')
        self.prototype_vectors['A'] = self.lib.get('A')

        # Enforce minimum similarity constraint on other prototypes
        base = self.prototype_vectors['A']
        D = self.dimensions
        self.prototype_vectors['B'] = self.normalize(base+2*np.random.randn(D))
        self.prototype_vectors['C'] = self.normalize(base+2*np.random.randn(D))

        # Generate random label vectors
        for l in labels:
            self.lib.add(l)
            self.label_vectors[l] = self.lib.get(l)

        # Generate training stimuli vectors
        for p in prototypes:
            for n in range(num_train):
                vec = self.prototype_vectors[p]
                self.train_vectors[p+'T'+str(n)] = self.disturb(vec, dval*r)

        # Generate testing stimuli vectors
        high, low, old = dict(), dict(), dict()

        for p in prototypes:
            ind = random.sample(range(num_train), num_per_level)
            for n in range(num_per_level):
                vec = self.prototype_vectors[p]
                high[p+'H'+str(n)] = self.disturb(vec, dval*r)
                low[p+'L'+str(n)] = self.disturb(vec, dval)
                old[p+'T'+str(ind[n])] = self.train_vectors[p+'T'+str(ind[n])]
            
        # Merge test vectors into a single dictionary
        self.test_vectors = dict(high.items() + low.items() + old.items())
        self.test_vectors.update(self.prototype_vectors)

        # Normalize all of the test vectors that have been produced
        for vec in self.test_vectors:
            self.test_vectors[vec] = self.normalize(self.test_vectors[vec])

        # Create string description of memory for SPA parser
        self.memory = 'AT0*LabelA+AT1*LabelA+AT2*LabelA+AT3*LabelA+' +\
                      'BT0*LabelB+BT1*LabelB+BT2*LabelB+BT3*LabelB+' +\
                      'CT0*LabelC+CT1*LabelC+CT2*LabelC+CT3*LabelC' 

        self.output = np.zeros((1, self.dimensions))

    def get_label(self, stimulus):
        if 'A' in stimulus:
            category = 'LabelA'
        elif 'B' in stimulus:
            category = 'LabelB'
        else:
            category = 'LabelC'
        return category

class Brooks(Stimuli):
    """
    Describes objects that generate and handle stimuli for Regehr and Brooks
    experiment involving exemplar-based categorization. This object serves 
    as container for the stimuli used in the experiment. There are two 
    versions of the experiment involving rules and exemplars, respectively.
    The rule methods below label the stimuli according to one of four rules
    and generate generic feature representations that encode the content of 
    each rule.

    Parameters:
    -----------
    dimensions : int
        The dimensionality of the stimuli vectors to be generated.
    dval : float
        The standard deviation of the gaussian distribution sampled from to
        'distort' the prototype vectors when generating different classes of
        stimuli. More specifically, the parameter sets the STD of the
        distribution corresponding to the 'low' distortion stimuli, while a 
        fixed multiplier is used to generate the STD of the distribution 
        corresponding to the 'high' distortion stimuli. 
    seed : int
        The seed for the random number generator. Setting this allows for the
        replication of stimuli across runs of the model.
    """
    def __init__(self, dimensions, dval, seed=None):
        self.task = 'BROOKS'
        self.feature_name_vectors = dict()
        self.pos_value_vectors = dict()
        self.neg_value_vectors = dict()
        self.label_vectors = dict()
        self.stimuli_A = dict()
        self.stimuli_B = dict()
        self.labelled_stimuli = dict()
        self.dimensions = dimensions
        self.dval = dval
        self.lib = Library()

        # Set seed for random number generator
        if seed != None:
            np.random.seed(seed)

        # Analytic stimuli structures from the experiment
        subset_a = [[0, 0, 0, 0, 0], [0, 0, 0, 1, 1], [0, 1, 1, 0, 1], 
                    [0, 1, 1, 1, 0], [1, 0, 1, 0, 1], [1, 0, 1, 1, 0], 
                    [1, 1, 0, 0, 0], [1, 1, 0, 1, 1]]
        subset_b = [[0, 0, 1, 0, 0], [0, 0, 1, 1, 1], [0, 1, 0, 0, 1], 
                    [0, 1, 0, 1, 0], [1, 0, 0, 0, 1], [1, 0, 0, 1, 0], 
                    [1, 1, 1, 0, 0], [1, 1, 1, 1, 1]]

        # Features used in all the possible stimuli structures
        feature_names = ['BODY', 'NECK', 'SPOTS', 'LEGS_NUM', 'LEGS_LEN']
        neg_values = ['ROUND', 'SHORT_NECK', 'NO', 'TWO','SHORT_LEGS']
        pos_values = ['ANGULAR', 'LONG_NECK', 'YES', 'SIX', 'LONG_LEGS']

        labels = ['Builder', 'Digger']

        # Generate feature name vectors
        for f in feature_names:
            self.lib.add(f)
            self.feature_name_vectors[f] = self.lib.get(f)
    
        # Generate feature value vectors
        for nv in neg_values:
            self.lib.add(nv)
            self.neg_value_vectors[nv] = self.lib.get(nv)
        for pv in pos_values:
            self.lib.add(pv)
            self.pos_value_vectors[pv] = self.lib.get(pv)

        # Generate label vectors
        for l in labels:
            self.lib.add(l)
            self.label_vectors[l] = self.lib.get(l)

        # Build stimulus sets
        tempA, tempB = np.zeros(self.dimensions), np.zeros(self.dimensions)

        for i in range(len(subset_a)):
            counter = 0    
            for j in range(len(feature_names)):    
                fvec = self.feature_name_vectors[feature_names[counter]]
                pvec = self.pos_value_vectors[pos_values[counter]]  
                nvec = self.neg_value_vectors[neg_values[counter]]  

                # Apply specified distortion to feature vectors
                pos_value_vector = self.disturb(pvec, self.dval)
                neg_value_vector = self.disturb(nvec, self.dval)    
                
                # Bind feature value vectors to feature name vectors and sum
                if subset_a[i][j] == 1:
                    tempA += self.convolve(fvec, pos_value_vector)
                elif subset_a[i][j] == 0:
                    tempA += self.convolve(fvec, neg_value_vector)
                if subset_b[i][j] == 1:
                    tempB += self.convolve(fvec, pos_value_vector)
                elif subset_b[i][j] == 0:
                    tempB += self.convolve(fvec, neg_value_vector)
                counter += 1

            self.stimuli_A['A'+str(i)] = tempA
            self.stimuli_B['B'+str(i)] = tempB
            tempA, tempB = np.zeros(self.dimensions), np.zeros(self.dimensions) 

        # Normalize all of the generated stimuli
        for stim in self.stimuli_A:
            self.stimuli_A[stim] = self.normalize(self.stimuli_A[stim])
        for stim in self.stimuli_B:
            self.stimuli_B[stim] = self.normalize(self.stimuli_B[stim])
        
        self.output = np.zeros((1, self.dimensions))
        self.test_vectors = dict(self.stimuli_A.items()+self.stimuli_B.items())

    def rule1(self):
        # Indices for items that are builders according to this rule
        self.A = [2,4,5,7]
        self.B = [1,4,6,7]
        # Indices for subset B items that are BT according to this rule
        self.BT = [1,2,5,6]
        self.assign_labels()

        # Build exemplar memory description according to this rule for SPA
        self.memory = 'A0*Digger+A1*Digger+A2*Builder+A3*Digger+'\
                      'A4*Builder+A5*Builder+A6*Digger+A7*Builder'

    def rule2(self):
        # Indices for items that are builders according to this rule
        self.A = [2,3,5,6]
        self.B = [0,3,6,7]
        # Indices for subset B items that are BT according to this rule
        self.BT = [0,2,5,7]
        self.assign_labels()

        # Build exemplar memory description according to this rule for SPA
        self.memory = 'A0*Digger+A1*Digger+A2*Builder+A3*Builder'\
                      '+A4*Digger+A5*Builder+A6*Builder+A7*Digger'

    def rule3(self):
        # Indices for items that are builders according to this rule
        self.A = [3,4,5,7]
        self.B = [1,5,6,7]
        # Indices for subset B items that are BT according to this rule
        self.BT = [1,3,4,6]
        self.assign_labels()

        # Build exemplar memory description according to this rule for SPA
        self.memory = 'A0*Digger+A1*Digger+A2*Digger+A3*Builder+'\
                      'A4*Builder+A5*Builder+A6*Digger+A7*Builder'

    def rule4(self):
        # Indices for items that are builders according to this rule
        self.A = [2,3,4,6]
        self.B = [0,2,6,7]
        # Indices for subset B items that are BT according to this rule
        self.BT = [0,3,4,7]
        self.assign_labels()

        # Build exemplar memory description according to this rule for SPA
        self.memory = 'A0*Digger+A1*Digger+A2*Builder+A3*Builder'\
                      '+A4*Builder+A5*Digger+A6*Builder+A7*Digger'

    def assign_labels(self):
        # Assign correct labels to all of the stimuli
        for stim in self.stimuli_A:
            if len([x for x in self.A if str(x) in stim]) > 0:
                self.labelled_stimuli[stim] = ['Builder', 'Training Item']
            else:
                self.labelled_stimuli[stim] = ['Digger', 'Training Item']

        for stim in self.stimuli_B:
            if (len([x for x in self.B if str(x) in stim]) > 0) and \
               (len([x for x in self.BT if str(x) in stim]) > 0):
                self.labelled_stimuli[stim] = ['Builder', 'Bad Transfer']
            elif (len([x for x in self.B if str(x) in stim]) > 0) and \
                 (len([x for x in self.BT if str(x) in stim]) == 0):
                self.labelled_stimuli[stim] = ['Builder', 'Good Transfer']
            elif (len([x for x in self.B if str(x) in stim]) == 0) and \
                 (len([x for x in self.BT if str(x) in stim]) > 0):
                self.labelled_stimuli[stim] = ['Digger', 'Bad Transfer']
            else:
                self.labelled_stimuli[stim] = ['Digger', 'Good Transfer']

    def get_label(self, stimulus):
        label = self.labelled_stimuli[stimulus][0]
        return label

class Murphy(Stimuli):
    """
    Describes objects that generate and handle stimuli for the Lin and Murphy
    experiment involving knowledge-based categorization. This object serves 
    as container for the stimuli used in the experiment.

    Parameters:
    -----------
    dimensions : int
        The dimensionality of the stimuli vectors to be generated.
    dval : float
        The standard deviation of the gaussian distribution sampled from to
        'distort' the prototype vectors when generating different classes of
        stimuli. More specifically, the parameter sets the STD of the
        distribution corresponding to the 'low' distortion stimuli, while a 
        fixed multiplier is used to generate the STD of the distribution 
        corresponding to the 'high' distortion stimuli. 
    seed : int
        The seed for the random number generator. Setting this allows for the
        replication of stimuli across runs of the model.
    """
    def __init__(self, dimensions, dval, seed=None):
        self.task = 'MURPHY'
        self.features = dict()
        self.stimuli = dict()
        self.label_vectors = dict()
        self.dimensions = dimensions
        self.lib = Library()

        # Set seed for random number generator
        if seed != None:
            np.random.seed(seed)        

        stimuli_names = ['Prototype', 'ConsistentA', 'ConsistentB', 'Control']
        feature_names = ['A','B','C','D']
        labels = ['YES', 'NO']

        self.structure = [[1,1,1,1],[1,0,1,1],[0,1,1,1],[0,0,0,1]]

        # Generate random feature vectors for each of 8 categories
        for f in feature_names:
            for i in range(8): 
                key = f+str(i)       
                self.lib.add(key)
                self.features[key] = self.lib.get(key) 

        for l in labels:
            self.lib.add(l)
            self.label_vectors[l] = self.lib.get(l)
        
        # Generate 3 noisy stimuli per stimulus type for all 8 categories
        for n in stimuli_names:
            for i in range(8):
                key = n+str(i)
                self.stimuli[key] = np.zeros(self.dimensions)
                count = 0
                for val in self.structure[stimuli_names.index(n)]:
                    vec = self.features[feature_names[count]+str(i)]
                    self.stimuli[key] += val * self.disturb(vec, dval)
                    count += 1

        self.output = np.zeros((1, self.dimensions))
        self.output[:] = self.label_vectors['YES']

    def get_label(self, stimulus):
        label = 'YES'
        return label

class Library(Stimuli):
    """ 
    A library of labelled sitmuli vectors. The library is a dictionary of 
    label: vector pairs, and all vectors are generated to ensure an upper
    bound on their degree of similarity with one another.

    Parameters
    ----------
    max_similarity : float
        The maximum permitted cosine value between any two vectors in the lib
    """
    def __init__(self, max_similarity=0.15):
        self.lib = dict()
        self.dimensions = 128
        self.max_similarity = max_similarity

    def add(self, key):
        if key in self.lib.keys():
            print "Library already contains this item!"
        else:
            self.lib[key] = self.generate_vector()

    def get(self, key):
        if key not in self.lib.keys():
            print "Library does not contain this item!"
        else:
            return self.lib[key]

    def generate_vector(self):
        count = 0
        indicator = 0
        p = self.normalize(np.random.randn(self.dimensions))              
        while count<10000:
            if len(self.lib.keys()) > 0:
                for vector in self.lib:
                    similarity = np.dot(self.lib[vector], p)
                    if similarity > self.max_similarity:
                        indicator += 1
                if indicator > 0:
                    indicator = 0
                    p = self.normalize(np.random.randn(self.dimensions))
                    count += 1
                    continue
                else:
                    break
            else:
                break
        if count >= 10000:
            print 'Warning: Could not create a semantic pointer'+\
                  ' within similarity constraints'
        return p
