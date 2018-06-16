import pandas
import decide
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

DATA = '../data/iris.csv'
df = pandas.read_csv(DATA)
X = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']
y = 'Species'

model = decide.Model(
	df,
	X,
	y,
	decision_tree = DecisionTreeClassifier(max_depth=5),
	random_forest = RandomForestClassifier(max_depth=5, min_samples_leaf=3),
)

sample = [4.8, 2.5, 3.4, 1.1]
model.predict(sample)