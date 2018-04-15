import sys, os
import time
import numpy as np
import tensorflow as tf

from urllib.request import urlretrieve
from os.path import isfile, isdir
from tqdm import tqdm
import zipfile

from collections import Counter
import random
import argparse

import mutils


def write_embeddings(embed_mat, int_to_vocab, filename):
    embed_dim = embed_mat.shape[1]
    total_vocab = embed_mat.shape[0]
    fmt = '{} ' + '{:.6f} ' * embed_dim + '\n'
    with open(filename, mode='w') as f:
        for i in range(total_vocab):
            f.write(fmt.format(int_to_vocab[i], *embed_mat[i]))
    return

# ## Making batches
def get_target(words, idx, window_size=5):
    ''' Get a list of words in a window around an index. '''
    
    R = np.random.randint(1, window_size+1)
    start = idx - R if (idx - R) > 0 else 0
    stop = idx + R
    target_words = set(words[start:idx] + words[idx+1:stop+1])
    
    return list(target_words)


def get_batches(words, batch_size, window_size=5):
    ''' Create a generator of word batches as a tuple (inputs, targets) '''
    
    n_batches = len(words)//batch_size
    
    # only full batches
    words = words[:n_batches*batch_size]
    
    for idx in range(0, len(words), batch_size):
        x, y = [], []
        batch = words[idx:idx+batch_size]
        for ii in range(len(batch)):
            batch_x = batch[ii]
            batch_y = get_target(batch, ii, window_size)
            y.extend(batch_y)
            x.extend([batch_x]*len(batch_y))
        yield x, y
    

def get_text(clean=False, targets='targets.txt'):
    """ load texts
    """
    text = ''

    files = []
    with open('./data/' + targets) as f:
        for line in f:
            files.append(line.rstrip('\n'))

    for target in files:
        print('Processing file:', target)
        with open('./data/' + target) as f:
            for line in f:
                cleaned = line.strip().rstrip("\n\r")
                if clean:
                    cleaned = mutils.clean(cleaned)
                text += ' ' + cleaned

    print('Final text size:',type(text), len(text))
    print('First 500 words:', text[:500])
    print('Last  500 words:', text[-500:])
    return text



def main():
    parser = argparse.ArgumentParser(description="Skipgram POS embedding code")
    parser.add_argument("--data_dir", default="data/")
    parser.add_argument("--model_dir", default="models/checkpoints/")
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--min_freq", type=int, default=10)
    parser.add_argument("--embed_dim", type=int, default=300)
    parser.add_argument("--n_sample", help='n negative sample', type=int, default=30)
    parser.add_argument("--model_name", type=str, default='skipgram_pos')
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--target_file", type=str, default='targets.txt')
    parser.add_argument("--exp", help='experiment number', type=int, default=0)
    parser.add_argument('--seed', help='use 1 to generate random seed or int', type=int, default=1)
    parser.add_argument('--clean', help='use 1 to clean', type=int, default=0)

    args = parser.parse_args()
    print('Args:', args)

    # check to make sure necessary directories exist
    if not os.path.exists(args.data_dir):
        sys.exit('Error: create data directory')
    if not os.path.exists(args.model_dir):
        sys.exit('Error: create checkpoint directory')

    if args.clean == 0:
        clean = False
    else: 
        clean = True 

    # load text sources
    text = get_text(clean, args.target_file)

    # ## Preprocessing
    # NB that words have a natural order as the occur in language and have context..that we'll use later!
    words = mutils.preprocess(text, args.min_freq)
    # NB notice that these are just strings..so we can preprocess our text to any specification we wish!
    print('First 500 processed words:', words[:500])
    print('Last  500 processed words:', words[-500:])
    print("Total words:  {}".format(len(words)))
    print("Unique words: {}".format(len(set(words))))

    # the dictionary my be ordered by frequency but the int_word order is same as text
    vocab_to_int, int_to_vocab = mutils.create_lookup_tables(words)
    int_words = [vocab_to_int[word] for word in words]

    # ### Special vocabulary terms we'd like to track
    vocab_terms = ['VBG','NNS','JJ','IN', 'DT', 'PRP']
    print('Vocab terms:')
    for i in vocab_terms:
        print('term:', i, 'vocab integer:\t', vocab_to_int[i])
    
    print()
    vocab_terms_ints = [vocab_to_int[i] for i in vocab_terms]

    # ## Subsampling
    threshold = 1e-5
    word_counts = Counter(int_words)
    total_count = len(int_words)
    freqs = {word: count/total_count for word, count in word_counts.items()}
    p_drop = {word: 1 - np.sqrt(threshold/freqs[word]) for word in word_counts}
    train_words = [word for word in int_words if random.random() < (1 - p_drop[word])]

    train_graph = tf.Graph()
    with train_graph.as_default():
        inputs = tf.placeholder(tf.int32, [None], name='inputs')
        labels = tf.placeholder(tf.int32, [None, None], name='labels')

    # ## Embedding
    n_vocab = len(int_to_vocab)
    n_embedding = args.embed_dim # Number of embedding features 
    with train_graph.as_default():
        embedding = tf.Variable(tf.random_uniform((n_vocab, n_embedding), -1, 1))
        embed = tf.nn.embedding_lookup(embedding, inputs)

    # Number of negative labels to sample
    n_sampled = args.n_sample
    with train_graph.as_default():
        softmax_w = tf.Variable(tf.truncated_normal((n_vocab, n_embedding), stddev=0.1))
        softmax_b = tf.Variable(tf.zeros(n_vocab))
    
        # Calculate the loss using negative sampling
        loss = tf.nn.sampled_softmax_loss(softmax_w, softmax_b, 
                                      labels, embed,
                                      n_sampled, n_vocab)
    
        cost = tf.reduce_mean(loss)
        optimizer = tf.train.AdamOptimizer().minimize(cost)


    # ## Validation
    with train_graph.as_default():
        ## From Thushan Ganegedara's implementation
        valid_size = 16 # Random set of words to evaluate similarity on.
        valid_window = 20
        # pick 8 samples from (0,100) and (1000,1100) each ranges. lower id implies more frequent 
        valid_examples = np.array(random.sample(range(valid_window), valid_size//2))
        valid_examples = np.append(valid_examples, 
                               random.sample(range(25,15+valid_window), valid_size//2))
        # add our special terms here
        valid_examples = np.append(valid_examples, vocab_terms_ints)
    
        valid_size = valid_size + len(vocab_terms_ints)
        valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
    
        # We use the cosine distance:
        norm = tf.sqrt(tf.reduce_sum(tf.square(embedding), 1, keep_dims=True))
        normalized_embedding = embedding / norm
        valid_embedding = tf.nn.embedding_lookup(normalized_embedding, valid_dataset)
        similarity = tf.matmul(valid_embedding, tf.transpose(normalized_embedding))


    epochs = args.epochs
    batch_size = args.batch_size
    window_size = 10

    with train_graph.as_default():
        saver = tf.train.Saver()

    with tf.Session(graph=train_graph) as sess:
        iteration = 1
        loss = 0
        sess.run(tf.global_variables_initializer())
    
        for e in range(1, epochs+1):
            batches = get_batches(train_words, batch_size, window_size)
            start = time.time()
            for x, y in batches:
                
                feed = {inputs: x,
                        labels: np.array(y)[:, None]}
                train_loss, _ = sess.run([cost, optimizer], feed_dict=feed)
                
                loss += train_loss
                
                if iteration % 100 == 0: 
                    end = time.time()
                    print("Epoch {}/{}".format(e, epochs),
                          "Iteration: {}".format(iteration),
                          "Avg. Training loss: {:.4f}".format(loss/100),
                          "{:.4f} sec/batch".format((end-start)/100))
                    loss = 0
                    start = time.time()
                
                if iteration % 1000 == 0:
                    # note that this is expensive (~20% slowdown if computed every 500 steps)
                    sim = similarity.eval()
                    for i in range(valid_size):
                        valid_word = int_to_vocab[valid_examples[i]]
                        top_k = 6 # number of nearest neighbors
                        nearest = (-sim[i, :]).argsort()[1:top_k+1]
                        log = 'Nearest to %s:' % valid_word
                        for k in range(top_k):
                            close_word = int_to_vocab[nearest[k]]
                            log = '%s %s,' % (log, close_word)
                        print(log)
                
                iteration += 1
        save_path = saver.save(sess, args.model_dir + args.model_name + ".ckpt")
        embed_mat = sess.run(normalized_embedding)

    # Restore the trained network if you need to:
    with train_graph.as_default():
        saver = tf.train.Saver()

    with tf.Session(graph=train_graph) as sess:
        saver.restore(sess, tf.train.latest_checkpoint(args.model_dir))
        embed_mat = sess.run(embedding)

    write_embeddings(embed_mat, int_to_vocab, args.data_dir +
                     args.model_name + '_' + str(args.embed_dim) +
                     '_embeddings_exp_' + str(args.exp) + '_epoch_' +
                     str(args.epochs) + '.txt')
    return


if __name__ == "__main__":
    main()
