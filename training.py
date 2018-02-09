import tensorflow.python.platform

import numpy as np
import tensorflow as tf

# Global variables.
NUM_LABELS = 11    # The number of labels.
BATCH_SIZE = 400  # The number of training examples to use per training step.

# Define the flags useable from the command line.
tf.app.flags.DEFINE_string('train', None,
                           'File containing the training data (labels & features).')
tf.app.flags.DEFINE_integer('num_epochs', 1,
                            'Number of examples to separate from the training '
                            'data for the validation set.')
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
        labels.append(int(float(row[256])))

    # Convert the array of float arrays into a numpy float matrix.
    fvecs_np = np.matrix(fvecs).astype(np.float32)

    # Convert the array of int labels into a numpy array.
    labels_np = np.array(labels).astype(dtype=np.uint8)

    # Convert the int numpy array into a peaks number matrix.
    labels_peaks = (np.arange(NUM_LABELS) == labels_np[:, None]).astype(np.float32)

    # Return a pair of the feature matrix and the peaks number label matrix.
    return fvecs_np,labels_peaks

def main(argv=None):
    # Be verbose?
    verbose = FLAGS.verbose
    
    # Get the data.
    train_data_filename = FLAGS.train
    
    # Extract it into numpy matrices.
    train_data,train_labels = extract_data(train_data_filename)
    
    # Get the shape of the training data.
    train_size,num_features = train_data.shape

    # Get the number of epochs for training.
    num_epochs = FLAGS.num_epochs

    # This is where training samples and labels are fed to the graph.
    # These placeholder nodes will be fed a batch of training data at each
    # training step using the {feed_dict} argument to the Run() call below.
    x = tf.placeholder("float", shape=[None, num_features])
    y_ = tf.placeholder("float", shape=[None, NUM_LABELS])
    
    # Define and initialize the network.
    
    

    # These are the weights that inform how much each feature contributes to
    # the classification.
    W = tf.Variable(tf.zeros([num_features,NUM_LABELS]))
    b = tf.Variable(tf.zeros([NUM_LABELS]))
    y = tf.nn.softmax(tf.matmul(x,W) + b)
    
    tf.add_to_collection('coll', x);
    tf.add_to_collection('coll', y_);
    tf.add_to_collection('coll', W);
    tf.add_to_collection('coll', b);

    # Optimization.
    cross_entropy = -tf.reduce_sum(y_*tf.log(y))
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

    # Evaluation.
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    
    tf.add_to_collection('coll', accuracy);

    # Create a local session to run this computation.
    with tf.Session() as s:
        # Run all the initializers to prepare the trainable parameters.
        tf.initialize_all_variables().run()
        if verbose:
            print 'Initialized!'
            print
            print 'Training.'

        # Iterate and train.
        for step in xrange(num_epochs * train_size // BATCH_SIZE):
            if verbose:
                print step,
                
            offset = (step * BATCH_SIZE) % train_size
            batch_data = train_data[offset:(offset + BATCH_SIZE), :]
            batch_labels = train_labels[offset:(offset + BATCH_SIZE)]
            train_step.run(feed_dict={x: batch_data, y_: batch_labels})
            
            if verbose and offset >= train_size-BATCH_SIZE:
                print
                
        # Export the model to saved_model.meta.
        # Creates a saver.
        saver0 = tf.train.Saver()
        saver0.save(s, './saved_model')
        # Generates MetaGraphDef.
        saver0.export_meta_graph('./saved_model.meta')

        # Give very detailed output.
        if verbose:
            print "Applying model to samples."
            print 
            first = train_data[:1]
            #print "Point =", first #first = test_data[:1]
            #print "Wx+b = ", s.run(tf.matmul(first,W)+b)
            #print "softmax(Wx+b) = ", s.run(tf.nn.softmax(tf.matmul(first,W)+b))
            s.run(W)
            s.run(b)
            print "softmax: ", s.run(tf.nn.softmax(tf.matmul(first,W)+b))
            
        print "Accuracy:", accuracy.eval(feed_dict={x: train_data, y_: train_labels})

    
if __name__ == '__main__':
    tf.app.run()
