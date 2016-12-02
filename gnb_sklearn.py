import numpy as np
import util as util
from sklearn.naive_bayes import GaussianNB

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

model = GaussianNB()
model = model.fit(train_dataset, train_labels)

print(model.score(train_dataset, train_labels))
print(model.score(test_dataset, test_labels))