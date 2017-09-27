import pandas as pd
import numpy as np

import pylab as pl
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.model_selection import train_test_split
from sklearn import metrics
import itertools
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn import neighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import label_binarize

import plot_utils

def SMOTE_binary(Xtrain,Ytrain):
	"""
	Use SMOTE to oversample the minority class.
	"""
	sm = SMOTE(random_state=13, ratio = 1.0)
	X_train_res, y_train_res = sm.fit_sample(Xtrain, Ytrain)
	return X_train_res, y_train_res

def SMOTE_multiclass(Xtrain,Ytrain):
	"""
	Use SMOTE to oversample the minority class.
	"""
	for i in np.arange(len(set(Ytrain))-1):
		sm = SMOTE(random_state=13, ratio = 1.0)
		Xtrain, Ytrain = sm.fit_sample(Xtrain, Ytrain)
	return Xtrain,Ytrain

def LR_binary_lambdaselection(X_train,y_train,X_valid,y_valid,lambda_list):
	"""
	Logistic Regression classifier for purchace intent binary prediction.
	Input: input feature matrix
	Output: 
		print metrics
		plot confusion matrix
		save model to current directory
	"""
	for i in range(len(lambda_list)):
		c = lambda_list[i]
		print(c)
		lr = LogisticRegression(penalty = 'l2', C = c)
		lr.fit(X_train,y_train)
		probs = lr.predict_proba(X_valid)
		predicted = lr.predict(X_valid)

		# Compute Precision-Recall and plot curve
		precision, recall, thresholds = precision_recall_curve(y_valid, probs[:, 1])
		area = auc(recall, precision)
		print("Area Under Curve: %0.2f" % area)

		plot_utils.prcurve_binary(precision,recall,area)




def LR_binary_traintest(X_train,y_train,X_test,y_test,c):
	"""
	Logistic Regression classifier for purchace intent binary prediction.
	Input: input feature matrix
	Output: 
		print metrics
		plot confusion matrix
		save model to current directory
	"""
	lr = LogisticRegression(penalty = 'l2', C = c)
	lr.fit(X_train,y_train)
	probs = lr.predict_proba(X_test)
	predicted = lr.predict(X_test)

	print('accuracy:',metrics.accuracy_score(y_test, predicted))
	print('precision:',metrics.precision_score(y_test, predicted))
	print('recall:',metrics.recall_score(y_test, predicted))
	print('f1:', metrics.f1_score(y_test, predicted))

	#plot confusion matrix
	class_names = ['no intent','intent']    

	# Compute confusion matrix
	cnf_matrix = metrics.confusion_matrix(y_test, predicted)
	np.set_printoptions(precision=2)

	# Plot normalized confusion matrix
	plt.figure()
	plot_utils.plot_confusion_matrix_normalized(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

	# Plot non-normalized confusion matrix
	plt.figure()
	plot_utils.plot_confusion_matrix_unnormalized(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

	plt.show()





def kNN_multiclass_kselection(X_train,y_train,X_valid,y_valid,k_list):
	"""
	k-Nearest Neighbor classifier for 7-class marketing funnel stage prediction.
	Input: input feature matrix; k value;
	Output: 
		print metrics
		plot confusion matrix
		save model to current directory
	"""
	yvalid_binary = label_binarize(y_valid, classes=[-2,-1,0, 1, 2,3,4])

	weights = 'distance'
	for j in range(len(k_list)):
		n_neighbors = k_list[j]
		print(n_neighbors)
		clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
		clf.fit(X_train, y_train)
		predicted = clf.predict(X_valid)
		probs = clf.predict_proba(X_valid)

		n_classes = 7
		colors = itertools.cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal','red','green'])
		# Compute Precision-Recall and plot curve
		precision = dict()
		recall = dict()
		thresholds = dict()
		average_precision = dict()
		for i in range(n_classes):
			precision[i], recall[i], thresholds[i] = precision_recall_curve(yvalid_binary[:,i],probs[:, i])
			average_precision[i] = metrics.average_precision_score(yvalid_binary[:,i], probs[:, i])

		# Compute micro-average ROC curve and ROC area
		precision["micro"], recall["micro"], thresholds['micro'] = precision_recall_curve(yvalid_binary.ravel(),probs.ravel())
		average_precision["micro"] = metrics.average_precision_score(yvalid_binary, probs,average="micro")

		plot_utils.prcurve_multiclass(precision,recall,average_precision,n_classes,colors)



def kNN_multiclass_traintest(X_train,y_train,X_test,y_test,k):
	"""
	k-Nearest Neighbor classifier for 7-class marketing funnel stage prediction.
	Input: input feature matrix; k value;
	Output: 
		print metrics
		plot confusion matrix
		save model to current directory
	"""
	weights = 'distance'
	clf = neighbors.KNeighborsClassifier(k, weights=weights)
	clf.fit(X_train, y_train)
	predicted = clf.predict(X_test)
	probs = clf.predict_proba(X_test)

	ytest_binary = label_binarize(y_test, classes=[-2,-1,0, 1, 2,3,4])
	predicted_binary = label_binarize(predicted, classes=[-2,-1,0, 1, 2,3,4])

	print('average precision score aka AUC PR curve:',metrics.average_precision_score(ytest_binary, predicted_binary,average='weighted'))
	print('accuracy:',metrics.accuracy_score(y_test,predicted))
	print(metrics.classification_report(y_test,predicted))

	#plot confusion matrix
	class_names = ['spam','neg','neu','interest','evaluation','intent','purchased']    

	# Compute confusion matrix
	cnf_matrix = metrics.confusion_matrix(y_test, predicted)
	np.set_printoptions(precision=2)

	# Plot normalized confusion matrix
	plt.figure()
	plot_utils.plot_confusion_matrix_normalized(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

	# Plot non-normalized confusion matrix
	plt.figure()
	plot_utils.plot_confusion_matrix_unnormalized(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

	plt.show()

def RF_binary_gridsearch(X_train,y_train,X_valid,y_valid,numTrees_list,maxDepth_list):
	"""
	Random Forest classifier for purchace intent binary prediction.
	Input: input feature matrix; number of trees as a list; max tree depth as a list;
	Output: 
		print metrics
		plot confusion matrix
		save model to current directory
	"""
	for i in range(len(numTrees_list)):
		for j in range(len(maxDepth_list)):
			num_trees = numTrees_list[i]
			tree_depth = maxDepth_list[j]
			print(num_trees, tree_depth)
			clf = RandomForestClassifier(n_estimators=num_trees,max_depth=tree_depth)
			clf.fit(X_train,y_train)
			probs = clf.predict_proba(X_valid)#/ X_train_normfactor
			predicted = clf.predict(X_valid)#/ X_train_normfactor

			# Compute Precision-Recall and plot curve
			precision, recall, thresholds = precision_recall_curve(y_valid, probs[:, 1])
			area = auc(recall, precision)
			print("Area Under Curve: %0.2f" % area)

			plot_utils.prcurve_binary(precision,recall,area)


def RF_binary_traintest(X_train,y_train,X_test,y_test,numTrees,maxDepth):
	"""
	Random Forest classifier for purchace intent binary prediction.
	Input: input feature matrix; number of trees; max tree depth;
	Output: 
		print metrics
		plot confusion matrix
		save model to current directory
	"""
	clf = RandomForestClassifier(n_estimators=numTrees,max_depth=maxDepth)
	clf.fit(X_train,y_train)
	print('feature importance:',clf.feature_importances_)

	probs = clf.predict_proba(X_test)#/ X_train_normfactor
	predicted = clf.predict(X_test)#/ X_train_normfactor
	print('accuracy:',metrics.accuracy_score(y_test, predicted))
	print('precision:',metrics.precision_score(y_test, predicted))
	print('recall:',metrics.recall_score(y_test, predicted))
	print('f1:', metrics.f1_score(y_test, predicted))

	#plot confusion matrix
	class_names = ['no intent','intent']    

	# Compute confusion matrix
	cnf_matrix = metrics.confusion_matrix(y_test, predicted)
	np.set_printoptions(precision=2)

	# Plot normalized confusion matrix
	plt.figure()
	plot_utils.plot_confusion_matrix_normalized(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

	# Plot non-normalized confusion matrix
	plt.figure()
	plot_utils.plot_confusion_matrix_unnormalized(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

	plt.show()




def RF_multiclass_gridsearch(X_train,y_train,X_valid,y_valid,numTrees_list,maxDepth_list):
	"""
	Random Forest classifier for 7-class marketing funnel stage prediction.
	Input: input feature matrix; number of trees as a list; max tree depth as a list;
	Output: 
		print metrics
		plot confusion matrix
		save model to current directory
	"""
	yvalid_binary = label_binarize(y_valid, classes=[-2,-1,0, 1, 2,3,4])

	for k in range(len(numTrees_list)):
		for j in range(len(maxDepth_list)):
			num_trees = numTrees_list[k]
			tree_depth = maxDepth_list[j]
			print(num_trees, tree_depth)
			clf = RandomForestClassifier(n_estimators=num_trees,max_depth=tree_depth)
			clf.fit(X_train,y_train)
			probs = clf.predict_proba(X_valid)#/ X_train_normfactor
			predicted = clf.predict(X_valid)#/ X_train_normfactor
			
			n_classes = 7
			colors = itertools.cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal','red','green'])
			# Compute Precision-Recall and plot curve
			precision = dict()
			recall = dict()
			thresholds = dict()
			average_precision = dict()
			for i in range(n_classes):
				precision[i], recall[i], thresholds[i] = precision_recall_curve(yvalid_binary[:,i],probs[:, i])
				average_precision[i] = metrics.average_precision_score(yvalid_binary[:,i], probs[:, i])

			# Compute micro-average ROC curve and ROC area
			precision["micro"], recall["micro"], thresholds['micro'] = precision_recall_curve(yvalid_binary.ravel(),probs.ravel())
			average_precision["micro"] = metrics.average_precision_score(yvalid_binary, probs,average="micro")

			plot_utils.prcurve_multiclass(precision,recall,average_precision,n_classes,colors)



def RF_multiclass_traintest(X_train,y_train,X_test,y_test,numTrees,maxDepth):
	"""
	Random Forest classifier for 7-class marketing funnel stage prediction.
	Input: input feature matrix; number of trees; max tree depth;
	Output: 
		print metrics
		plot confusion matrix
		save model to current directory
	"""
	clf = RandomForestClassifier(n_estimators=numTrees,max_depth=maxDepth)
	clf.fit(X_train,y_train)
	probs = clf.predict_proba(X_test)#/ X_train_normfactor
	predicted = clf.predict(X_test)#/ X_train_normfactor

	print('feature importance:',clf.feature_importances_)

	ytest_binary = label_binarize(y_test, classes=[-2,-1,0, 1, 2,3,4])
	predicted_binary = label_binarize(predicted, classes=[-2,-1,0, 1, 2,3,4])

	print('average precision score aka AUC PR curve:',metrics.average_precision_score(ytest_binary, predicted_binary,average='weighted'))
	print('accuracy:',metrics.accuracy_score(y_test,predicted))
	print(metrics.classification_report(y_test,predicted))

	#plot confusion matrix
	class_names = ['spam','neg','neu','interest','evaluation','intent','purchased']    

	# Compute confusion matrix
	cnf_matrix = metrics.confusion_matrix(y_test, predicted)
	np.set_printoptions(precision=2)

	# Plot normalized confusion matrix
	plt.figure()
	plot_utils.plot_confusion_matrix_normalized(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

	# Plot non-normalized confusion matrix
	plt.figure()
	plot_utils.plot_confusion_matrix_unnormalized(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

	plt.show()


