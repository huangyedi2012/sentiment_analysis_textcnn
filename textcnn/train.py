#! /usr/bin/env python
# coding=utf-8

import datetime
import os
import time

import tensorflow as tf

import data_input_helper as data_helpers
from text_cnn import TextCNN
# Data loading params
from textcnn import tokenization
from textcnn.metrics import get_binary_metrics

tf.flags.DEFINE_string("train_data_file", "../data/train.csv", "Data source for the positive data.")
tf.flags.DEFINE_string("test_data_file", "../data/test.csv", "Data source for the positive data.")
tf.flags.DEFINE_string("w2v_file", "/data/workspace/data/embeddings/sgns.wiki.bigram-char", "word2vec file path")
tf.flags.DEFINE_string("class_file", "../data/class.txt", "word2vec file path")

# Model Hyperparameters
tf.flags.DEFINE_integer("seq_len", 256, "Max length of sentence (default: 256)")
tf.flags.DEFINE_string("filter_sizes", "2,3,4", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 10, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 50, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS


def pad_sequence(tokens, seq_len):
    if len(tokens) > seq_len:
        return tokens[:seq_len]
    else:
        tokens.extend(['<PAD>'] * (seq_len - len(tokens)))
        return tokens


def load_data(filepath):
    print("Loading data...")
    texts, labels = data_helpers.load_data_and_labels(filepath)
    seq_len = FLAGS.seq_len
    if seq_len == -1:
        seq_len = max([len(x) for x in texts])
    print('len(x) = ', len(texts))
    print('max_document_length = ', seq_len)

    tokens = [pad_sequence(tokenizer.tokenize(text), seq_len) for text in texts]
    x = [tokenizer.convert_tokens_to_ids(token) for token in tokens]
    y = [int(label) for label in labels]
    return x, y, seq_len


def train():
    x_train, y_train, seq_len = load_data(FLAGS.train_data_file)
    x_test, y_test, seq_len = load_data(FLAGS.test_data_file)
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = TextCNN(
                word2vec,
                seq_len=seq_len,
                num_classes=len(labels),
                vocab_size=len(vocab),
                embedding_size=dims,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                l2_reg_lambda=FLAGS.l2_reg_lambda)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-5)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            def train_step(x_batch, y_batch):
                """
                A single training step
                """
                feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch,
                    cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                # _, step, summaries, loss, accuracy,(w,idx) = sess.run(
                #     [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy,cnn.get_w2v_W()],
                #     feed_dict)
                _, step, loss, predictions = sess.run(
                    [train_op, global_step, cnn.loss, cnn.predictions],
                    feed_dict)

                time_str = datetime.datetime.now().isoformat()
                acc, precision, recall, f_beta = get_binary_metrics(pred_y=predictions, true_y=y_batch)
                print("{}: step {}, loss {:g}, precision {:g}, recall:{:g}".format(time_str, step, loss, acc, precision, recall))

            def dev_step(x_batch, y_batch, writer=None):
                """
                Evaluates model on a dev set
                """
                feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch,
                    cnn.dropout_keep_prob: 1.0
                }
                step, loss, predictions = sess.run(
                    [global_step, cnn.loss, cnn.predictions],
                    feed_dict)
                acc, precision, recall, f_beta = get_binary_metrics(pred_y=predictions, true_y=y_batch)
                print("test evaluations: loss {:g}, precision {:g}, recall:{:g}".format(loss, acc, precision, recall))

            # Generate batches
            batches = data_helpers.batch_iter(list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)

            def dev_test():
                # batches_dev = data_helpers.batch_iter(list(zip(x_test, y_test)), FLAGS.batch_size, 1)
                batches_dev = data_helpers.batch_iter(list(zip(x_test, y_test)), len(x_test), 1)
                for batch_dev in batches_dev:
                    x_batch_dev, y_batch_dev = zip(*batch_dev)
                    dev_step(x_batch_dev, y_batch_dev)

            # Training loop. For each batch...
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                # Training loop. For each batch...
                if current_step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    dev_test()

                if current_step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))


def read_class_label(class_file):
    labels = []
    with open(class_file, 'r', encoding='utf-8') as fr:
        for label in fr:
            labels.append(label.strip())
    id2label = {idx: label for idx, label in enumerate(labels)}
    label2id = {label: idx for idx, label in enumerate(labels)}
    label_one_hot = {k: [0] * len(labels) for k in labels}
    for label in label_one_hot:
        one_hot = label_one_hot[label]
        one_hot[label2id[label]] = 1
        label_one_hot[label] = one_hot
    return labels, id2label, label2id, label_one_hot


if __name__ == "__main__":
    vocab, word2vec, dims = data_helpers.load_word_vector(FLAGS.w2v_file)
    labels, id2label, label2id, label_one_hot = read_class_label(FLAGS.class_file)
    tokenizer = tokenization.FullTokenizer(vocab)
    train()
