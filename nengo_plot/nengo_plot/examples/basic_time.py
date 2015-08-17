import nengo
import numpy as np

model = nengo.Network()
with model:
    input = nengo.Node(lambda t: np.sin(6*t))
    a = nengo.Ensemble(n_neurons=100, dimensions=1)
    nengo.Connection(input, a)
    pIn = nengo.Probe(input)
    pA = nengo.Probe(a)
    pA_spikes = nengo.Probe(a, 'spikes')

sim = nengo.Simulator(model)
sim.run(1)

import nengo_plot
plot = nengo_plot.Time(sim.trange())
plot.add('input', sim.data[pIn])
plot.add('a', sim.data[pA])

# do a raw spike raster
plot.add_spikes('a', sim.data[pA_spikes])

# do a spike raster with only the 64 most varying neurons, and sorted
#  (clustered) by similarity
plot.add_spikes('a', sim.data[pA_spikes], sample_by_variance=64, cluster=True)
plot.save('basic_time.png')
plot.show()


