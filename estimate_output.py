import tensorflow.python.platform

import numpy as np
import tensorflow as tf

# Global variables.
NUM_LABELS = 11    # The number of labels.

# Define the flags useable from the command line.
tf.app.flags.DEFINE_string('test', None,
                           'File containing the test data (labels & features).')
tf.app.flags.DEFINE_boolean('verbose', False, 'Produce verbose output.')
FLAGS = tf.app.flags.FLAGS

# Extract numpy representations of the labels and features given rows consisting of:
#   label, feat_0, feat_1, ..., feat_n
def extract_data(filename):

    # Arrays to hold the labels and feature vectors.
    labels = []
    fvecs = []

    # Iterate over the rows, splitting the label from the features. Convert labels
    # to integers and features to floats.
    for line in file(filename):
        row = line.split(",")
        fvecs.append([float(x) for x in row[0:255]])
        labels.append(int(float(row[256]))) #before int(row[256])

    # Convert the array of float arrays into a numpy float matrix.
    fvecs_np = np.matrix(fvecs).astype(np.float32)

    # Convert the array of int labels into a numpy array.
    labels_np = np.array(labels).astype(dtype=np.uint8)

    # Convert the int numpy array into a peaks number matrix.
    labels_peaks = (np.arange(NUM_LABELS) == labels_np[:, None]).astype(np.float32)

    # Return a pair of the feature matrix and the one-hot label matrix.
    return fvecs_np,labels_peaks

def main(argv=None):
    # Be verbose?
    verbose = FLAGS.verbose
    
    # Get the data.
    test_data_filename = FLAGS.test
    
    # Extract it into numpy matrices.
    test_data, test_labels = extract_data(test_data_filename)

    # Create a local session to run this computation.
    with tf.Session() as s:
        # Load variables
        new_saver = tf.train.import_meta_graph('./saved_model.meta')
        new_saver.restore(s, './saved_model')
        # Addes loss and train.
        x = tf.get_collection('coll')[0]
        y_ = tf.get_collection('coll')[1]
        W = tf.get_collection('coll')[2]
        b = tf.get_collection('coll')[3]
        accuracy = tf.get_collection('coll')[4]
        print W
        print b

        # Give very detailed output.
        if verbose:
            print "Applying model to tests."
            print 
            first_ten = test_data[:10]
            s.run(W)
            s.run(b)
            print "softmax: ", s.run(tf.nn.softmax(tf.matmul(first_ten,W)+b))
        
        print "Accuracy:", accuracy.eval(feed_dict={x: test_data, y_: test_labels})

    
if __name__ == '__main__':
    tf.app.run()
