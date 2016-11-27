import numpy as np
import tensorflow as tf
import util as util

data, labels, feature_size = util.getDataAndLabelsV2()

graph = tf.Graph()
with graph.as_default():
	# parameters
	beta = 0.001
	theta = 0.5
	batch_size = 128
	labels_size = len(labels)
	valid_start_index = labels_size - 2 * batch_size
	valid_end_index = labels_size - batch_size
	
	tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, feature_size))
	tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, 1))

	tf_valid_dataset = tf.constant(np.array(data[valid_start_index:valid_end_index]), tf.float32)
	tf_valid_labels = tf.constant(np.array(labels[valid_start_index:valid_end_index]), tf.float32)

	tf_test_dataset = tf.constant(np.array(data[valid_end_index:]), tf.float32)
	tf_test_labels = tf.constant(np.array(labels[valid_end_index:]), tf.float32)

	w = tf.Variable(tf.truncated_normal([feature_size, 1]))
	b = tf.Variable(tf.zeros([1]))

	def model(data1):
		"""
		model itself
		"""
		return tf.matmul(data1, w) + b

	# training
	logits = model(tf_train_dataset)
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
	regularized_loss = tf.nn.l2_loss(w)
	total_loss = loss + beta * regularized_loss

	# optimizer
	optimizer = tf.train.GradientDescentOptimizer(theta).minimize(total_loss)

	# predictions
	train_prediction = tf.nn.softmax(logits)
	valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
	test_prediction = tf.nn.softmax(model(tf_test_dataset))

def accuracy(predictions, labels):
	#print predictions
	#print labels
	#np_p = np.array(list(predictions))
	#np_l = np.array(list(labels))
	wrong = np.count_nonzero(np.diff(np.array([predictions.transpose(), labels]), axis=0))
	print wrong
	total = len(predictions.transpose()[0])
	print total
	return 100.0 * (total - wrong) / total
	#return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape(0))

num_steps = labels_size / batch_size - 2
with tf.Session(graph=graph) as session:
	tf.initialize_all_variables().run()
	print("Initialized")
	for step in range(num_steps):

		batch_data = data[step*batch_size:step*batch_size+batch_size]
		batch_labels = np.array([labels[step*batch_size:step*batch_size+batch_size]]).transpose()

		feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
		
		_, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)

		if (step % 10 == 0) :
			print("Minibatch loss at step %d: %f" % (step, l))
			print("Minibatch accuracy: %.lf%%" % accuracy(predictions, batch_labels.transpose()))
			print("Validation accuracy: %.lf%%" % accuracy(valid_prediction.eval(), np.array([tf_valid_labels.eval()])))
	print("Test accuracy: %.lf%%" % accuracy(test_prediction.eval(), np.array([tf_test_labels.eval()])))
