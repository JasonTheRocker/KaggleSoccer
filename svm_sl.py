from sklearn import svm
from sklearn.metrics import accuracy_score
import numpy as np
import util as util
from matplotlib import pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

# Global variables.
batch_size = 2000  # The number of training examples to use per training step.
beta = 0.5
theta = 0.1
num_epochs = 1
svmC = 0.01

_data, labels = util.getDataAndLabels_svm()
labels_size = len(labels)
valid_start_index = labels_size - 2 * batch_size
valid_end_index = labels_size - batch_size
#print data[0]
def PCA(data, chunk_size):
	data_ = []
	num_features = len(data[0])
	last_start = (num_features / chunk_size) * chunk_size
	for i in range(len(data)):
		each = []
		for j in range(num_features / chunk_size):
			each.append(np.mean(data[i][j*chunk_size:j*chunk_size+chunk_size]))
		each.append(np.mean(data[i][last_start:]))
		data_.append(each)
	return data_

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

if __name__ == '__main__':
	data = PCA(_data, 40)
	train_data = data[:valid_start_index]
	train_label = labels[:valid_start_index]

	valid_data = data[valid_start_index:valid_end_index]
	valid_labels = labels[valid_start_index:valid_end_index]

	test_data = data[valid_end_index:]
	test_labels = labels[valid_end_index:]

	# Get the shape of the training data.
	train_size,num_features = labels_size - 2*batch_size, len(data[0])

	#clf = svm.SVC(kernel='rbf')
	#clf.fit(train_data, train_label)
	#print clf

	# title = "Learning Curves (SVM, RBF kernel, $\gamma=0.001$)"
	# cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
	# plot_learning_curve(clf, title, train_data, train_label, (0.0, 1.01), cv=cv, n_jobs=4)
	# plt.show()
	
	# print "rbf SVC:"
	# print accuracy_score(train_label, clf.predict(train_data))
	# print accuracy_score(valid_labels, clf.predict(valid_data))
	# print accuracy_score(test_labels, clf.predict(test_data))

	clf = svm.SVC(kernel='linear')
	clf.fit(train_data, train_label)
	#print clf

	title = "Learning Curves (SVM, linear kernel)"
	#cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
	plot_learning_curve(clf, title, train_data, train_label, (0.0, 1.01), n_jobs=4)
	plt.show()

	# SVC is more expensive so we do a lower number of CV iterations:
	
	

	



