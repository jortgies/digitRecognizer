import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from PIL import Image

'''
    downloading the MNIST-dataset of handwritten digits
    http://yann.lecun.com/exdb/mnist/
    
    y labels are oh-encoded that means:
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0] equals 3 
    
    The images are 28x28 pixels each and are flattened to 
    1D vectors of size 784.  
'''

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
n_train = mnist.train.num_examples  # 55,000
n_validation = mnist.validation.num_examples  # 5000
n_test = mnist.test.num_examples  # 10,000

# DEFINING NEURONAL NETWORK
n_input = 784  # input layer (28x28 pixels)
n_hidden1 = 512  # 1st hidden layer
n_hidden2 = 256  # 2nd hidden layer
n_hidden3 = 128  # 3rd hidden layer
n_output = 10  # output layer (0-9 digits)

# these values are set initially and remain constant throughout the process

'''
The learning rate represents how much the parameters will adjust at each step of the learning process
Larger learning rates can converge faster, but also have the potential to overshoot the optimal 
values as they are updated.
'''
learning_rate = 1e-4

# how many times it will go through the training step
n_iterations = 5000

# how many training examples it is using at each step
batch_size = 128

'''
Threshold at which it eliminates some units at random.
Will be used in the final hidden layer to give each unit a 50% chance 
of being eliminated at every training step. This helps prevent overfitting.
'''
dropout = 0.5


# SET UP TENSORFLOW
'''
Set up the network as a computational graph for TensorFlow to execute.
he core concept of TensorFlow is the tensor, a data structure similar 
to an array or list. initialized, manipulated as they are passed through 
the graph, and updated through the learning process.
'''
# [None, 784]: 784 input neurons
# None represents any amount, as we will be feeding in an undefined number of 784-pixel images
X = tf.placeholder("float", [None, n_input])

# [None, 10] as we will be using it for an undefined number of label outputs,
# with 10 possible classes
Y = tf.placeholder("float", [None, n_output])

# This tensor is used to control the dropout rate, and we initialize it as a placeholder
# rather than an immutable variable because we want to use the same tensor both
# for training (when dropout is set to 0.5) and testing (when dropout is set to 1.0).
keep_prob = tf.placeholder(tf.float32)


# WEIGHTS
'''
As we know the initial value actually has a significant impact on the final accuracy of the model.
We'll use random values from a truncated normal distribution for the weights. 
We want them to be close to zero, so they can adjust in either a positive 
or negative direction, and slightly different, so they generate different errors.
'''
weights = {
    'w1': tf.Variable(tf.truncated_normal([n_input, n_hidden1], stddev=0.1)),
    'w2': tf.Variable(tf.truncated_normal([n_hidden1, n_hidden2], stddev=0.1)),
    'w3': tf.Variable(tf.truncated_normal([n_hidden2, n_hidden3], stddev=0.1)),
    'out': tf.Variable(tf.truncated_normal([n_hidden3, n_output], stddev=0.1)),
}


# BIASES
'''
We use a small constant value for the biases to ensure that the tensors activate in the
initial stages and therefore contribute to the propagation.
The weights and bias tensors are stored in dictionary objects for ease of access.
'''
biases = {
    'b1': tf.Variable(tf.constant(0.1, shape=[n_hidden1])),
    'b2': tf.Variable(tf.constant(0.1, shape=[n_hidden2])),
    'b3': tf.Variable(tf.constant(0.1, shape=[n_hidden3])),
    'out': tf.Variable(tf.constant(0.1, shape=[n_output]))
}

# OPERATIONS
'''
Each hidden layer will execute matrix multiplication on the previous layer’s outputs 
and the current layer’s weights, and add the bias to these values. 
At the last hidden layer, we will apply a dropout operation using our keep_prob value of 0.5.
'''
layer_1 = tf.add(tf.matmul(X, weights['w1']), biases['b1'])
layer_2 = tf.add(tf.matmul(layer_1, weights['w2']), biases['b2'])
layer_3 = tf.add(tf.matmul(layer_2, weights['w3']), biases['b3'])
# using rate=1-keep_prob instead of keep_prob
layer_drop = tf.nn.dropout(layer_3, rate=1-keep_prob)
output_layer = tf.matmul(layer_3, weights['out']) + biases['out']

# LOSS FUNCTION
'''
A popular choice of loss function in TensorFlow programs is cross-entropy, 
also known as log-loss, which quantifies the difference between two probability
distributions (the predictions and the labels).

We also need to choose the optimization algorithm which will be used to minimize
the loss function. A process named gradient descent optimization is a common method
for finding the (local) minimum of a function by taking iterative steps along the gradient
in a negative (descending) direction.

We will be using the Adam optimizer.
This extends upon gradient descent optimization by using momentum to speed up the process 
through computing an exponentially weighted average of the gradients and using that in the adjustments.
'''
# changed cross_entropy_with_logits to _v2
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=Y, logits=output_layer
        ))
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

# TRACK TRAINING
'''
Evaluating the accuracy.
Check that from the first iteration to the last, loss decreases and accuracy increases;
they will also allow us to track whether or not we have ran enough iterations to 
reach a consistent and optimal result.

In correct_pred, we use the arg_max function to compare which images are being 
predicted correctly by looking at the output_layer (predictions) and Y (labels),
and we use the equal function to return this as a list of Booleans. 
We can then cast this list to floats and calculate the mean to get a total accuracy score.
'''
correct_pred = tf.equal(tf.argmax(output_layer, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# FEED NETWORK WITH DATA
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# train on mini batches
for i in range(n_iterations):
    batch_x, batch_y = mnist.train.next_batch(batch_size)
    sess.run(train_step, feed_dict={
        X: batch_x, Y: batch_y, keep_prob: dropout
        })

    # print loss and accuracy (per minibatch)
    if i % batch_size == 0:
        minibatch_loss, minibatch_accuracy = sess.run(
            [cross_entropy, accuracy],
            feed_dict={X: batch_x, Y: batch_y, keep_prob: 1.0}
            )
        print(
            "Iteration",
            str(i),
            "\t| Loss =",
            str(minibatch_loss),
            "\t| Accuracy =",
            str(minibatch_accuracy)
            )

# TESTING
'''
Once the training is complete, we can run the session on the test images. 
This time we are using a keep_prob dropout rate of 1.0 to ensure all units 
are active in the testing process.
'''

test_accuracy = sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1.0})
print("\nAccuracy on test set:", test_accuracy)

# predict user input
img1 = np.invert(Image.open("userDigits/1.jpeg").convert('L')).ravel()
img3 = np.invert(Image.open("userDigits/3.jpeg").convert('L')).ravel()
img7 = np.invert(Image.open("userDigits/7.jpeg").convert('L')).ravel()
img8 = np.invert(Image.open("userDigits/8.jpeg").convert('L')).ravel()
prediction1 = sess.run(tf.argmax(output_layer, 1), feed_dict={X: [img1]})
prediction3 = sess.run(tf.argmax(output_layer, 1), feed_dict={X: [img3]})
prediction7 = sess.run(tf.argmax(output_layer, 1), feed_dict={X: [img7]})
prediction8 = sess.run(tf.argmax(output_layer, 1), feed_dict={X: [img8]})

print("Prediction for test image showing '1' :", np.squeeze(prediction1))
print("Prediction for test image showing '3' :", np.squeeze(prediction3))
print("Prediction for test image showing '7' :", np.squeeze(prediction7))
print("Prediction for test image showing '8' :", np.squeeze(prediction8))


