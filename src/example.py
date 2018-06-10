import pandas
import decide

DATA = '../data/iris.csv'
df = pandas.read_csv(DATA)
X = ['SepalWidth','SepalLength','PetalWidth','PetalLength']
y = 'Species'
model = decide.Model(df,X,y)
model.validate()
model.analyze(0.2)