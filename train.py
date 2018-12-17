import os
import datetime
import tensorflow as tf
import numpy as np
import tqdm
from utils import batch_generator, load_word_vectors
import data_loader

# params
NUM_CLASSES = 5
LSTM_NUM_UNITS = 128
D_KEEP_PROB = 0.5
DATA_BASE_DIR = "data"
WORD_VECTORS_PATH = "embeddings/word_vectors.npy"

def evaluate():
    """
    Given model & params - evaluate model's performance by
    running it on the evaluation set
    """


def process_data():
    """
    Load data & Process it, returning data that is ready for training / evaluating
    """


def train():
    """
    Build and Train model by given params
    """
    
    # params
    # T
    max_seq_length = None
    keep_prob = 0.5
    n_hidden = 128
    num_classes = 5
    learning_rate = 1e-3
    model_save_path = "models/"
    # How many iterations?
    iterations = 10000
    batch_size = 128
    word_vector_dim = 50 
    
    # ************** Pre-Model **************
    # Load data
    data_params = data_loader.get_data_params(DATA_BASE_DIR)
    max_seq_length = data_params["max_seq_length"]
    X_train, X_eval, y_train, y_eval = data_loader.load_data(data_params)
    
    # Load Glove / Word2Vec embbeding vectors
    word_vectors = load_word_vectors(WORD_VECTORS_PATH)
    
    # Batch generators
    train_batch_generator = batch_generator((X_train, y_train), batch_size)
    test_batch_generator = batch_generator((X_eval, y_eval), batch_size)

    # ************** Model **************
    # placeholders
    labels = tf.placeholder(tf.float32, [None, num_classes])
    input_data = tf.placeholder(tf.int32, [None, max_seq_length])
    
    # data processing
    data = tf.Variable(tf.zeros([batch_size, max_seq_length,
        word_vector_dim]), dtype=tf.float32)

    data = tf.nn.embedding_lookup(word_vectors, input_data)

    # lstm cell
    lstm_cell = tf.nn.rnn_cell.LSTMCell(n_hidden)
    lstm_cell = tf.nn.rnn_cell.DropoutWrapper(cell=lstm_cell, output_keep_prob=keep_prob)
    # Do we need the state tuple? Because we don't want out cell to bi
    # initialized with the state from previous sentence
    ## rnn_tuple_state = tf.nn.rnn_cell.LSTMStateTuple(init_state[0], init_state[1])

    outputs, _ = tf.nn.dynamic_rnn(lstm_cell, data, dtype=tf.float32)
    
    # output layer
    weight = tf.Variable(tf.truncated_normal([n_hidden, num_classes]))
    bias = tf.Variable(tf.constant(0.1, shape=[num_classes]))

    # Let's try this logic
    outputs = tf.transpose(outputs, [1, 0, 2])
    last = tf.gather(outputs, int(outputs.get_shape()[0]) - 1)
    prediction = (tf.matmul(last, weight) + bias)

    correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    # Metrics
    # Should we reduce_mean?
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=labels))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    
    # Summaries
    tf.summary.scalar('Loss', loss)
    tf.summary.scalar('Accuracy', accuracy)
    merged = tf.summary.merge_all()
    logdir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"

    
    
    # ************** Train **************
    print("Run 'tensorboard --logdir=./{}' to checkout tensorboard logs.".format(os.path.abspath(logdir)))
    print("==> training")
    
    # Train
    with tf.Session() as sess:
        writer = tf.summary.FileWriter(logdir, sess.graph)
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())

        # Py2.7 or Py3 (if 2.7 --> Change to xrange)
        for iteration in tqdm.tqdm(range(iterations)):
            # shoudn't get exception, but check this
            X, y = next(train_batch_generator)
            sess.run(optimizer, {input_data: X, labels: y})

            # Write summary
            if (iteration % 30 == 0):
                summary = sess.run(merged, {input_data: X, labels: y})
                writer.add_summary(summary, iteration)

            # Save the network every 10,000 iterations
            if (iteration % 10000 == 0 and iteration != 0):
                save_path = saver.save(sess, model_save_path, global_step=iteration)
                print("saved to %s" % save_path)
                
        writer.close()



def main():
    tf.reset_default_graph()    
    train()


if __name__ == '__main__':
    main()

    
