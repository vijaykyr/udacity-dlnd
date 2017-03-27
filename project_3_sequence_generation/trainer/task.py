
import argparse
import urllib2
import helper
import numpy as np
import problem_unittests as tests
from distutils.version import LooseVersion
import warnings
import tensorflow as tf
from tensorflow.contrib import seq2seq

with open('preprocess.p','wb') as output:
  output.write(urllib2.urlopen("https://storage.googleapis.com/vijays-sandbox-ml/simpsons/data/preprocess.p").read())
int_text, vocab_to_int, int_to_vocab, token_dict = helper.load_preprocess()

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer'
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
    
def get_inputs():
    """
    Create TF Placeholders for input, targets, and learning rate.
    :return: Tuple (input, targets, learning rate)
    """
    # TODO: Implement Function
    input = tf.placeholder(tf.int32,shape=(None,None),name="input")
    targets = tf.placeholder(tf.int32,shape=(None,None),name="targets")
    learning_rate = tf.placeholder(tf.float32,name="learning_rate")
    
    return input, targets, learning_rate

def get_init_cell(batch_size, rnn_size):
    """
    Create an RNN Cell and initialize it.
    :param batch_size: Size of batches
    :param rnn_size: Size of RNNs
    :return: Tuple (cell, initialize state)
    """
    # TODO: Implement Function
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    cell = tf.contrib.rnn.MultiRNNCell([lstm])
    initial_state = tf.identity(cell.zero_state(batch_size,tf.float32),name="initial_state")
    
    return cell, initial_state
  
def get_embed(input_data, vocab_size, embed_dim):
    """
    Create embedding for <input_data>.
    :param input_data: TF placeholder for text input.
    :param vocab_size: Number of words in vocabulary.
    :param embed_dim: Number of embedding dimensions
    :return: Embedded input.
    """
    # TODO: Implement Function
    return tf.contrib.layers.embed_sequence(input_data,vocab_size,embed_dim,
        initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32))

def build_rnn(cell, inputs):
    """
    Create a RNN using a RNN Cell
    :param cell: RNN Cell
    :param inputs: Input text data
    :return: Tuple (Outputs, Final State)
    """
    # TODO: Implement Function
    outputs, final_state = tf.nn.dynamic_rnn(cell, inputs,dtype=tf.float32)
    final_state = tf.identity(final_state,name="final_state")
    return outputs, final_state

def build_nn(cell, rnn_size, input_data, vocab_size):
    """
    Build part of the neural network
    :param cell: RNN cell
    :param rnn_size: Size of rnns
    :param input_data: Input data
    :param vocab_size: Vocabulary size
    :return: Tuple (Logits, FinalState)
    """
    # TODO: Implement Function
    inputs = get_embed(input_data, vocab_size, rnn_size)
    outputs, final_state = build_rnn(cell,inputs)
    
    #apply fully connected layer to output. automatically flattens input to 2d and expands result
    logits = tf.contrib.layers.fully_connected(outputs, vocab_size, activation_fn=None)
    
    return logits, final_state

def get_batches(int_text, batch_size, seq_length):
    """
    Return batches of input and target
    :param int_text: Text with the words replaced by their ids
    :param batch_size: The size of batch
    :param seq_length: The length of sequence
    :return: Batches as a Numpy array
    """
    # TODO: Implement Function
    n_batches = int(len(int_text) / (batch_size * seq_length))

    # Drop the last few characters to make only full batches
    xdata = np.array(int_text[: n_batches * batch_size * seq_length])
    ydata = np.array(int_text[1: n_batches * batch_size * seq_length + 1])

    x_batches = np.split(xdata.reshape(batch_size, -1), n_batches, 1)
    y_batches = np.split(ydata.reshape(batch_size, -1), n_batches, 1)

    return np.array(list(zip(x_batches, y_batches)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Input Arguments
    parser.add_argument(
      '--output_dir',
      help='GCS location to write checkpoints and export models',
      required=True
    )
    parser.add_argument(
        '--job-dir',
        help='this model ignores this field, but it is required by gcloud',
        default='junk'
    )
    args = parser.parse_args()
    arguments = args.__dict__
    output_dir = arguments.pop('output_dir')

    # Number of Epochs
    num_epochs = 1000
    # Batch Size
    batch_size = 128
    # RNN Size
    rnn_size = 512
    # Sequence Length
    seq_length = 70
    # Learning Rate
    learning_rate = .001
    # Show stats for every n number of batches
    show_every_n_batches = 50

    save_dir = './save'

    train_graph = tf.Graph()
    with train_graph.as_default():
        vocab_size = len(int_to_vocab)
        input_text, targets, lr = get_inputs()
        input_data_shape = tf.shape(input_text)
        cell, initial_state = get_init_cell(input_data_shape[0], rnn_size)
        logits, final_state = build_nn(cell, rnn_size, input_text, vocab_size)

        # Probabilities for generating words
        probs = tf.nn.softmax(logits, name='probs')

        # Loss function
        cost = seq2seq.sequence_loss(
            logits,
            targets,
            tf.ones([input_data_shape[0], input_data_shape[1]]))

        # Optimizer
        optimizer = tf.train.AdamOptimizer(lr)

        # Gradient Clipping
        gradients = optimizer.compute_gradients(cost)
        capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients]
        train_op = optimizer.apply_gradients(capped_gradients)

    batches = get_batches(int_text, batch_size, seq_length)

    with tf.Session(graph=train_graph) as sess:
        sess.run(tf.global_variables_initializer())

        for epoch_i in range(num_epochs):
            state = sess.run(initial_state, {input_text: batches[0][0]})

            for batch_i, (x, y) in enumerate(batches):
                feed = {
                    input_text: x,
                    targets: y,
                    initial_state: state,
                    lr: learning_rate}
                train_loss, state, _ = sess.run([cost, final_state, train_op], feed)

                # Show every <show_every_n_batches> batches
                if (epoch_i * len(batches) + batch_i) % show_every_n_batches == 0:
                    print('Epoch {:>3} Batch {:>4}/{}   train_loss = {:.3f}'.format(
                        epoch_i,
                        batch_i,
                        len(batches),
                        train_loss))

        # Save Model
        saver = tf.train.Saver()
        saver.save(sess, save_dir)
        print('Model Trained and Saved')