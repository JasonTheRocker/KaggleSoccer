import numpy as np
import tensorflow as tf
import util as util

data, labels = util.getDataAndLabels()
num_features = len(data[0])


graph = tf.Graph()
with graph.as_default():
	# parameters
	beta = 0.001
	theta = 0.5
	batch_size = 128
	labels_size = len(labels)
	valid_start_index = labels_size - 2 * batch_size
	valid_end_index = labels_size - batch_size
	
	x = tf.placeholder(tf.float32, [None, num_features])
	y_ = tf.placeholder(tf.float32, [None, 3])

	tf_valid_dataset = data[valid_start_index:valid_end_index]
	tf_valid_labels = labels[valid_start_index:valid_end_index]

	tf_test_dataset = tf.constant(np.array(data[valid_end_index:]), tf.float32)
	tf_test_labels = tf.constant(np.array(labels[valid_end_index:]), tf.float32)

	W = tf.Variable(tf.zeros([num_features,3]))
	b = tf.Variable(tf.zeros([3]))
	
	y = tf.matmul(x, W) + b
	
	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
	regularized_loss = tf.nn.l2_loss(W)
	total_loss = cross_entropy + beta * regularized_loss	
	
	train_step = tf.train.GradientDescentOptimizer(theta).minimize(total_loss)	



	# optimizer
	#optimizer = tf.train.GradientDescentOptimizer(theta).minimize(total_loss)

	# predictions
	#train_prediction = tf.nn.softmax(logits)
	#valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
	#test_prediction = tf.nn.softmax(model(tf_test_dataset))

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
		batch_labels = labels[step*batch_size:step*batch_size+batch_size]
		#batch_labels = np.array([labels[step*batch_size:step*batch_size+batch_size]]).transpose()

		feed_dict = {x : batch_data, y_ : batch_labels}
		
		#_, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
		_, l = session.run([train_step,total_loss], feed_dict=feed_dict)
		
		
		
		if (step % 5 == 0) :
			#taccuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(train_prediction,1), tf.argmax(batch_labels,1)), "float"))
			#vaccuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(valid_prediction,1), tf.argmax(tf_valid_labels,1)), "float"))
			correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
			accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
			print(session.run(accuracy, feed_dict={x: tf_valid_dataset,
						              y_: tf_valid_labels}))			
			
			print("Minibatch loss at step %d: %f" % (step, l))
			#print("Minibatch accuracy: %.lf%%" % accuracy(predictions, batch_labels.transpose()))
			#print("Validation accuracy: %.lf%%" % accuracy(valid_prediction.eval(), np.array([tf_valid_labels.eval()])))
	#print("Test accuracy: %.lf%%" % accuracy(test_prediction.eval(), np.array([tf_test_labels.eval()])))
