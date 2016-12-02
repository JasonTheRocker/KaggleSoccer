import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

def replaceStrFeature(word):
	new = word.replace("(","").replace(")","").replace("[","").replace("]","").replace("u'", "").replace("'", "").replace("00:00:00", "").replace("-","").replace("/","").strip()
	try:
		convert = float(new)
	except ValueError:
		if "high" in new:
			#convert = 1.0
			convert = 100.0
		elif "low" in new:
			#convert = -1.0
			convert = 50.0
		elif "le" in new:
			convert = 100.0
		elif "right" in new:
			convert = 50.0
		elif "None" in new:
			convert = 0.0
		elif "medium" in new:
			convert = 0.0
		elif "o" in new:
			convert = 0.0
		elif "y" in new:
			convert = 100.0
		elif "es" in new:
			convert = 0.0
		elif "ean" in new:
			convert = 0.0
		else:
			print new
			convert = new
		
	return convert

def replaceStrV2(w):
	new = word.replace("[","").replace("]","").strip()
	convert = float(new)
	return convert
	

def getDataAndLabels():
	f = open('FeaturesV1.txt', 'r')
	line = f.readline()
	labels = []
	data = []	
	feature_size = len(line.split(",")) - 2	

	while line != "":
		record = line.split(",")
		home = int(record[9].strip())
		away = int(record[10].strip())
		if home > away:
			labels.append([float(1),float(0),float(0)])
		elif home == away:
			labels.append([float(0),float(1),float(0)])
		else:
			labels.append([float(0),float(0),float(1)])
		record = map(replaceStrFeature, record)
		
		# need to pop out player id, home, away, 
		for i in range(56, 78):
			record.pop(55)
		for i in range(0, 11):
			record.pop(0)	
		for i in range(0, 22):
			record.pop(0)
		for i in range(0, 22):
			record.pop(0)		
			
		data.append(record)
		line = f.readline()
	f.close()
	return data, labels


def getDataAndLabels_svm():
	f = open('FeaturesV1.txt', 'r')
	line = f.readline()
	labels = []
	data = []	
	feature_size = len(line.split(",")) - 2	

	while line != "":
		record = line.split(",")
		home = int(record[9].strip())
		away = int(record[10].strip())
		if home > away:
			labels.append(1.0)
		elif home == away:
			labels.append(0.0)
		else:
			labels.append(-1.0)
		record = map(replaceStrFeature, record)
		
		# need to pop out player id, home, away, 
		for i in range(56, 78):
			record.pop(55)
		for i in range(0, 10):
			record.pop(0)	
		for i in range(0, 22):
			record.pop(0)
		for i in range(0, 22):
			record.pop(0)		
			
		data.append(record)
		line = f.readline()
	f.close()
	return data, labels

def PCAreduce(data):
	ret = []
	for i in data:
		b = [sum(i[current: current+40]) for current in xrange(0, len(i), 40)]
		ret.append(b)
	return ret


def getDataAndLabelsV2():
	f = open('FeaturesV2.txt', 'r')
	line = f.readline()
	labels = []
	data = []	
	feature_size = len(line.split(",")) - 2
	
	while line != "":
		record = line.split(",")
		record = map(replaceStrFeature, record)
		data.append(record)
		line = f.readline()
	f.close()
	
	f = open('Label.txt', 'r')
	line = f.readline()
	while line != "":
		record = float(line)
		labels.append(record)
		line = f.readline()
	f.close()	
	
	return data, labels, feature_size	

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

if __name__ == "__main__":
	data, labels = getDataAndLabels()
	f = open('Label.txt', 'w+')
	for i in labels:
        	print >>f,  i	
		