import numpy as np
import util as util
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

data, labels = util.getDataAndLabels()
data = util.PCAreduce(data)
num_features = len(data[0])

labels = np.argmax(labels, 1)

labels_size = len(labels)
valid_size = 2000
test_start_index = labels_size -  valid_size

train_dataset = data[:test_start_index]
train_labels = labels[:test_start_index]

test_dataset = data[test_start_index:]
test_labels = labels[test_start_index:]

model = LogisticRegression()
_model = model.fit(train_dataset, train_labels)

print(_model.score(train_dataset, train_labels))
print(_model.score(test_dataset, test_labels))

title = "Learning Curves (LR)"
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
util.plot_learning_curve(model, title, data, labels, (0.0, 1.01),cv=cv, n_jobs=3)
plt.show()