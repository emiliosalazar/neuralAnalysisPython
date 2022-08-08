# neuralAnalysisPython

In progress library for performing and tracking neural analyses (alongside git repository emiliosalazar/datajoint-sqlite).

Main classes:
- Dataset: meant for storing raw info about a neural dataset--trial segmentations, spike times, etc. Used to generate BinnedSpikeSet's given trial segmenting/binning parameters
- BinnedSpikeSet: subclassed numpy tensor with dimensions trial x channel/neuron x timepoint. When different trials have different numbers of timepoints, produces an object array. Underlying class methods appropriately deal with the distinction when necessary.
