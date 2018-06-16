import pandas
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn import metrics
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.model_selection import KFold, ShuffleSplit
import draw_tree
import lcplot
from matplotlib import pyplot
import os

OUTPUT_DIR = os.path.join(os.getcwd(), 'output')

#------------------------------------------------------------------------------
class Model(object):
	def __init__(self,
			df,
			X,
			y,
			logit 			= LogisticRegression(),
			decision_tree 	= DecisionTreeClassifier(),
			random_forest 	= RandomForestClassifier(),
		):
		self.df = df
		self.X = df[X]
		self.y = df[y]
		self.logit = logit
		self.dt = decision_tree
		self.rf = random_forest
		if not os.path.exists(OUTPUT_DIR):
			os.mkdir(OUTPUT_DIR)

	#--------------------------------------------------------------------------
	def validate(self):
		self._validate(self.logit)
		self._validate(self.dt)
		self._validate(self.rf)

	#--------------------------------------------------------------------------
	def _validate(self, cls):
		cv = KFold(n_splits=10, shuffle=True)

		#---------------------------------------------------------
		# Cross validate
		#---------------------------------------------------------
		print('\nCross validating', cls, 'with', cv)
		res = cross_validate(
			cls,
			self.X,
			self.y,
			scoring=['precision_weighted', 'recall_weighted'],
			cv=cv,
		)
		print('Precision: {}'.format(round(res['test_precision_weighted'].mean(),2)))
		print('Recall:    {}'.format(round(res['test_recall_weighted'].mean(),2)))

	#--------------------------------------------------------------------------
	def learning(self):
		#---------------------------------------------------------
		# Calculate learning curve
		#---------------------------------------------------------
		print('\nCalculate learning curve')
		output = os.path.join(OUTPUT_DIR, 'lc_logit.png')
		lcplot.plot(self.logit, self.X, self.y, title='Logit', output=output)
		output = os.path.join(OUTPUT_DIR, 'lc_decision_tree.png')
		lcplot.plot(self.dt, self.X, self.y, title='Decision Tree', output=output)
		output = os.path.join(OUTPUT_DIR, 'lc_random_forest.png')
		lcplot.plot(self.rf, self.X, self.y, title='Random Forest', output=output)

	#--------------------------------------------------------------------------
	def predict(self, sample):
		print('Sample:', sample, '\n')

		print('Estimate class probabilities using logistic regression')
		self.logit.fit(self.X,self.y)
		prob = self.logit.predict_proba([sample])[0]
		for i,c in enumerate(self.logit.classes_):
			print('Prob({}) = {}'.format(c, round(prob[i])), end='\t')
		print('\n')

		output_path = os.path.join(OUTPUT_DIR, 'path')
		self.dt.fit(self.X,self.y)
		path = self.dt.decision_path([sample])
		draw_tree.draw_path(self.dt, self.dt.classes_, self.X.columns, path.indices, output_path)
		print('Draw decision path using decision tree: {}.png'.format(output_path))
		print('Decision path contains red-bordered shapes.\n')

		print('Estimate class probabilities using random forest')
		self.rf.fit(self.X,self.y)
		prob = self.rf.predict_proba([sample])[0]
		for i,c in enumerate(self.rf.classes_):
			print('Prob({}) = {}'.format(c, round(prob[i])), end='\t')
		print('\n')

		print('Features of importance')
		display_width = max([len(f) for f in self.X.columns])
		for i, f in enumerate(self.X.columns):
			print('{:{w}}: {}'.format(f, round(self.rf.feature_importances_[i],4), w=display_width))
		print()

		for i, m in enumerate(self.rf.estimators_):
			output_path = os.path.join(OUTPUT_DIR, 'path_' + str(i))
			path = m.decision_path([sample])
			draw_tree.draw_path(m, self.rf.classes_, self.X.columns, path.indices, output_path)
			print('Draw decision for random tree {}: {}.png'.format(i, output_path))
		print()

		print('Default models')
		print(self.logit)
		print(self.dt)
		print(self.rf)
		print()

	#--------------------------------------------------------------------------
	def analyze(self, test_size=0.05):
		X_train, X_test, y_train, y_test = train_test_split(
			self.X,
			self.y,
			test_size=test_size,
		)
		self.logit.fit(X_train, y_train)

		#---------------------------------------------------------
		# Show precision/recall
		#---------------------------------------------------------
		print('\n(1) Classification result for different classes')
		y_pred = self.logit.predict(X_test)
		print(metrics.classification_report(y_test, y_pred))

		#---------------------------------------------------------
		# Examine probabilities
		#---------------------------------------------------------
		print('\n(2) Prediction probabilities of samples')
		y_prob = self.logit.predict_proba(X_test)

		for c in self.logit.classes_:
			print('{}'.format(c[0:6]), end='\t')
		print('ClassLabel     \tCorrect')
		values = []
		for i in range(len(y_prob)):
			v = [ round(p,2) for p in y_prob[i] ]
			v.append(y_test.values[i])
			v.append('Y' if y_test.values[i]==y_pred[i] else 'no')
			v.append(max(y_prob[i]))
			values.append(v)

		values.sort(key=lambda v: v[-1], reverse=True)

		n = len(self.logit.classes_)
		for v in values:
			for i in range(n):
				print(v[i], end='\t')
			print('{:10s}\t{}'.format(v[n], v[n+1]))

		#---------------------------------------------------------
		# Visualize decisions
		#---------------------------------------------------------
		print('\n(3) Visualize the decision making process')
		output_dt = os.path.join(OUTPUT_DIR, 'decision_tree')
		self.dt.fit(X_train, y_train)
		draw_tree.draw(self.dt, self.dt.classes_, X_train.columns, output_dt)
		print('Decision process is saved to {}.png'.format(output_dt))

		#---------------------------------------------------------
		# Compare random decision trees
		#---------------------------------------------------------
		print('\n(4) Compare decision trees in a random forest')
		self.rf.fit(X_train, y_train)
		for i, m in enumerate(self.rf.estimators_):
			output_rf = os.path.join(OUTPUT_DIR, 'random_tree_' + str(i))
			draw_tree.draw(m, self.rf.classes_, X_train.columns, output_rf)
			print('Decision process is saved to {}.png'.format(output_rf))

		print()

#------------------------------------------------------------------------------
