from graphgen.data.data import get_training_data

path = "/Users/bkalisetti658/desktop/graphgen/data/InteractionDR1.0/recorded_trackfiles/DR_USA_Roundabout_EP"

traces = get_training_data(1, path)

print(len(traces))

