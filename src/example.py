import pandas
import decide

DATA = '../data/iris.csv'
df = pandas.read_csv(DATA)
X = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']
y = 'Species'

model = decide.Model(df,X,y)

# Validate the model with logit, decision tree, and random forest
model.validate()

# Analyze a random sample consisting 20% of the data
model.analyze(0.2)

# Visualize learning curves
model.learning()