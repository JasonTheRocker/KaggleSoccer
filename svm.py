import tensorflow as tf
import numpy as np
import scipy.io as io
import util as util
from matplotlib import pyplot as plt

# Global variables.
batch_size = 200  # The number of training examples to use per training step.
beta = 0.5
theta = 0.1
# Define the flags useable from the command line.
tf.app.flags.DEFINE_string('train', None,
                           'File containing the training data (labels & features).')
tf.app.flags.DEFINE_integer('num_epochs', 1,
                            'Number of training epochs.')
tf.app.flags.DEFINE_float('svmC', 1,
                            'The C parameter of the SVM cost function.')
tf.app.flags.DEFINE_boolean('verbose', False, 'Produce verbose output.')
tf.app.flags.DEFINE_boolean('plot', True, 'Plot the final decision boundary on the data.')
FLAGS = tf.app.flags.FLAGS

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

#print PCA(data, 10)[0]
data = PCA(_data, 10)

def main(argv=None):
	verbose = FLAGS.verbose

	plot = FLAGS.plot



	valid_data = tf.constant(np.array(data[valid_start_index:valid_end_index]), tf.float32)
	valid_labels = tf.constant(np.array(labels[valid_start_index:valid_end_index]), tf.float32)

	test_data = tf.constant(np.array(data[valid_end_index:labels_size]), tf.float32)
	test_labels = tf.constant(np.array(labels[valid_end_index:labels_size]), tf.float32)

	# Get the shape of the training data.
	train_size,num_features = labels_size - 2*batch_size, len(data[0])

	# Get the number of epochs for training.
	num_epochs = FLAGS.num_epochs

	# Get the C param of SVM
	svmC = FLAGS.svmC

	# This is where training samples and labels are fed to the graph.
	# These placeholder nodes will be fed a batch of training data at each
	# training step using the {feed_dict} argument to the Run() call below.
	x = tf.placeholder("float", shape=[None, num_features])
	y = tf.placeholder("float", shape=[None,1])

	# Define and initialize the network.

	# These are the weights that inform how much each feature contributes to
	# the classification.
	W = tf.Variable(tf.zeros([num_features,1]))
	b = tf.Variable(tf.zeros([1]))
	y_raw = tf.matmul(x,W) + b

	# Optimization.
	regularization_loss = beta*tf.reduce_sum(tf.square(W)) 
	hinge_loss = tf.reduce_sum(tf.maximum(tf.zeros([batch_size,1]), 
	    1 - y*y_raw));
	svm_loss = regularization_loss + svmC*hinge_loss;
	train_step = tf.train.GradientDescentOptimizer(theta).minimize(svm_loss)

	# Evaluation.
	predicted_class = tf.sign(y_raw);
	correct_prediction = tf.equal(y,predicted_class)
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

	y_valid = tf.matmul(valid_data,W) + b
	valid = tf.sign(y_valid);
	correct_valid = tf.equal(valid_labels,valid)
	accuracy_valid = tf.reduce_mean(tf.cast(correct_valid, "float"))

	y_test = tf.matmul(test_data,W) + b
	test = tf.sign(y_test);
	correct_test = tf.equal(test_labels,test)
	accuracy_test = tf.reduce_mean(tf.cast(correct_test, "float"))

	# Create a local session to run this computation.
	with tf.Session() as s:
	    # Run all the initializers to prepare the trainable parameters.
	    tf.initialize_all_variables().run()
	    if verbose:
	        print 'Initialized!'
	        print
	        print 'Training.'

	    # Iterate and train.
	    for step in xrange(num_epochs * train_size / batch_size):
			if verbose:
			    print step,
			    
			offset = (step * batch_size) % train_size
			end = offset + batch_size
			batch_data = data[offset:end]
			batch_labels = labels[offset:end]
			train_step.run(feed_dict={x: batch_data, y: batch_labels})
			print 'loss: ', svm_loss.eval(feed_dict={x: batch_data, y: batch_labels})
			print "Accuracy on train:", accuracy.eval(feed_dict={x: batch_data, y: batch_labels})
			print "Accuracy on valid:", accuracy_valid.eval()

			if verbose and offset >= train_size-batch_size:
			    print
		#print "Accuracy on test:", accuracy_test.eval()

	    # Give very detailed output.
	    if verbose:
	        print
	        print 'Weight matrix.'
	        print s.run(W)
	        print
	        print 'Bias vector.'
	        print s.run(b)
	        print
	        print "Applying model to first test instance."
	        print
	        
		
		

if __name__ == '__main__':
    tf.app.run()
