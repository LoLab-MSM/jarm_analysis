from tropical.visualize_discretization import visualization_path
from jnk3_no_ask1 import model
import pickle

with open("dom_path_labels.pkl", "rb") as input_file:
    paths = pickle.load(input_file)

for keys, values in paths.items():
    visualization_path(model, values, 'path_{}.pdf'.format(keys))