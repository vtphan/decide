##-------------------------------------------------------------------------
# conda install python-graphviz
# Draw a decision tree:
# dot -Tpng -o output.png output.dot -Gdpi=200
##-------------------------------------------------------------------------
from sklearn.tree import export_graphviz
import os, subprocess
import sys
def visualize_tree(tree, feature_names, output):
	with open(output+".dot", 'w') as f:
		dot_data = export_graphviz(tree, out_file=f, feature_names=feature_names, filled=True, rounded=True)
	command = ["dot", "-Tpng", output+".dot", "-o", output + ".png", "-Gdpi=200"]
	if sys.platform.startswith('win'):
		subprocess.run(command, shell=True)
	else:
		subprocess.run(command)
