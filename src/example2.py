import pandas
import decide

DATA = '../data/iris.csv'
df = pandas.read_csv(DATA)
X = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']
y = 'Species'
model = decide.Model(df,X,y)

sample = [4.8, 2.5, 3.4, 1.1]
model.predict(sample)