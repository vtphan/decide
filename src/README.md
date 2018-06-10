Look at examply.py for how to use the module.

```
import pandas
import decide

DATA = '../data/iris.csv'
df = pandas.read_csv(DATA)
X = ['SepalWidth','SepalLength','PetalWidth','PetalLength']
y = 'Species'

model = decide.Model(df,X,y)

model.validate()		# Validate the model with logit, decision tree, and random forest

model.analyze(0.2)  	# Analyze a random 20% sample of the data
```

Comments:

+ decide.Model takes as input a Pandas dataframe.

+ The features, X, should be column names of the data frame.

+ The features, X, should not have categorical variables.  If there are categorical variables, they should be converted to binary variables first (e.g. using pandas.get_dummies).