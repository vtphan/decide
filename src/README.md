Look at examply.py for how to use the module.

```
import pandas
import decide

DATA = '../data/iris.csv'
df = pandas.read_csv(DATA)
X = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']
y = 'Species'
model = decide.Model(df, X, y)

sample = [4.8, 2.5, 3.4, 1.1]
model.predict(sample)
```

Comments:

+ decide.Model takes as input a Pandas dataframe.

+ The features, X, should be column names of the data frame.

+ The features, X, should not have categorical variables.  If there are categorical variables, they should be converted to binary variables first (e.g. using pandas.get_dummies).

Output:
```
Sample: [4.8, 2.5, 3.4, 1.1]

Estimate class probabilities using logistic regression
Prob(setosa) = 0.0	Prob(versicolor) = 1.0	Prob(virginica) = 0.0

Draw decision path using decision tree: ./output/path.png
Decision path contains red-bordered shapes.

Estimate class probabilities using random forest
Prob(setosa) = 0.0	Prob(versicolor) = 1.0	Prob(virginica) = 0.0

Features of importance
SepalLength: 0.0897
SepalWidth : 0.0126
PetalLength: 0.363
PetalWidth : 0.5347

Draw decision for random tree 0: ./output/path_0.png
Draw decision for random tree 1: ./output/path_1.png
Draw decision for random tree 2: ./output/path_2.png
Draw decision for random tree 3: ./output/path_3.png
Draw decision for random tree 4: ./output/path_4.png
Draw decision for random tree 5: ./output/path_5.png
Draw decision for random tree 6: ./path_6.png
Draw decision for random tree 7: ./output/path_7.png
Draw decision for random tree 8: ./output/path_8.png
Draw decision for random tree 9: ./output/path_9.png

Default models
LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
          penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
          verbose=0, warm_start=False)
DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
            max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, presort=False, random_state=None,
            splitter='best')
RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)

```