import collections
import math
import random
import datetime as dt
import numpy as np
import tensorflow as tf
import nltk
import pickle
print("modules imported")
path_output = r"/output/"
path_input = r""
def read_data(filename):
    with open(filename) as f:
        print("tokenizing")
        data = f.read().split() # for memory efficient tokenization.Takes a while
        print("Tokenization complete.")
    return data


def build_dataset(words, nmbr_words = 10000, take_all_words = False):
    """Process raw inputs into a dataset."""
    count = [['UNK', -1]]
    print("Counting words...")
    counts = collections.Counter(words)
    if take_all_words:
        nmbr_words = len(counts) # take all words into account
    count.extend(counts.most_common(nmbr_words - 1))
    del counts
    print("Words counted. Initializing Vocbulary.")
    vocab = []
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
        #vocab.append(word)
    print("Sorting vocab...")
    vocab.sort()
    print("Vocab Sorted. Top {} words taken as vocabulary".format(nmbr_words))
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
        #if word in set(vocab):
            index = dictionary[word]
            #index = vocab.index(word)
        else:
            index = 0  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    print("Data has been populated.")
    count[0][1] = unk_count
    del count
    count = []
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count,dictionary, reversed_dictionary


def build_pre_saved_dataset(words, nmbr_words = 18000):
    count = [['UNK', -1]]
    #counts = collections.Counter(words)
    #count.extend(counts.most_common(nmbr_words - 1))
    # load pickled dictionary
    print("Loading pre saved state...")
    with open(path_input+"dictionary.pkl" , 'rb') as f:
        dictionary = pickle.load(f)
    # load data
    with open(path_input+"data.pkl", 'rb') as f:
        data = pickle.load(f)
    # load embeddings
    with open(path_input+"final_embeddings.pkl" ,'rb') as f:
        embeddings = pickle.load(f)
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    print("All set. Ready to begin trianing.")
    return data, count, dictionary, reversed_dictionary, embeddings

    
def collect_pre_saved_data():
    print("Loading training data...")
    vocabulary = read_data(path_input+"new_train_data_cleaned.txt")
    print("Training data loaded.")
    data, count, dictionary, reverse_dictionary, embeddings = build_pre_saved_dataset(vocabulary)
    del vocabulary
    return data, count, dictionary, reverse_dictionary, embeddings


def collect_data(vocabulary_size=10000, take_all_words=False):
    vocabulary = read_data(path_input+"new_train_data_cleaned.txt")
    data, count, dictionary, reversed_dictionary = build_dataset(vocabulary,
                                        nmbr_words =  vocabulary_size ,
                                         take_all_words = take_all_words)
    print("Dataset Built. Pickling important data.")
    del vocabulary  # to reduce memory.
    # Pickle some for later use
    with open(path_output+"dictionary.pkl",'wb') as file:
        pickle.dump(dictionary, file)
    """with open("vocab.pkl",'wb') as file:
        pickle.dump(vocab, file)"""
    with open(path_output+"data.pkl",'wb') as file:
        pickle.dump(data, file)
    return data, count, dictionary, reversed_dictionary
    #return data, count, vocab


data_index = 0
# generate batch data
def generate_batch(data, batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    context = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window input_word skip_window ]
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    for i in range(batch_size // num_skips):
        target = skip_window  # input word at the center of the buffer
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]  # this is the input word
            context[i * num_skips + j, 0] = buffer[target]  # these are the context words
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    # Backtrack a little bit to avoid skipping words in the end of a batch
    data_index = (data_index + len(data) - span) % len(data)
    return batch, context


resume_training = True
take_all_words = False
vocabulary_size = 18000
if not resume_training:
    data, count, dictionary, reverse_dictionary = collect_data(vocabulary_size=vocabulary_size)
else:
    data, count, dictionary, reverse_dictionary, embeddings = collect_pre_saved_data()

if not take_all_words:
    total_words = vocabulary_size
else:
    total_words = len(set(dictionary))
batch_size = 128
embedding_size = 300  # Dimension of the embedding vector.
skip_window = 2     # How many words to consider left and right.
num_skips = 2        # How many times to reuse an input to generate a label.

# Pick a random validation set to sample nearest neighbors.
validation_size = 16     # Random set of words to evaluate similarity on.
validation_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(validation_window, validation_size, replace=False)
num_sampled = 64    # Number of negative examples to sample.

# Now the placeholders and then the model
graph = tf.Graph()
with graph.as_default():
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
    if resume_training:
        init = tf.constant(embeddings)
        embeddings = tf.get_variable('embeddings', initializer=init)
    else:
        embeddings = tf.Variable(tf.random_uniform([total_words, embedding_size], -1.0, 1.0))
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)
    weights = tf.Variable(tf.truncated_normal([total_words, embedding_size],
                          stddev=1.0 / math.sqrt(embedding_size)))
    biases = tf.Variable(tf.zeros([total_words]))
    hidden_out = tf.matmul(embed, tf.transpose(weights)) + biases
    # convert train_context to a one-hot format
    train_one_hot = tf.one_hot(train_labels, total_words)
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hidden_out, 
        labels=train_one_hot))
    # Construct the SGD optimizer using a learning rate of 1.0.
    #optimizer = tf.train.GradientDescentOptimizer(3.0).minimize(cross_entropy)
    optimizer = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(
          normalized_embeddings, valid_dataset)
    similarity = tf.matmul(
          valid_embeddings, normalized_embeddings, transpose_b=True)
    # Add variable initializer.
    init = tf.global_variables_initializer()


def save_embeddings(embeddings, normalized = False):
    if normalized:
        with open("final_normalized_embeddings.pkl" , 'wb') as file:
            pickle.dump(embeddings, file)
    else:
        with open("final_embeddings.pkl" , 'wb') as file:
            pickle.dump(embeddings, file)
        

def run(graph, num_steps):
    with tf.Session(graph=graph) as session:
      # We must initialize all variables before we use them.
      init.run()
      print('Initialized')
      average_loss = 0
      for step in range(num_steps):
        batch_inputs, batch_context = generate_batch(data,
            batch_size, num_skips, skip_window)
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_context}
        _, loss_val = session.run([optimizer, cross_entropy], feed_dict=feed_dict)
        average_loss += loss_val
        if step % 1000 == 0:
          if step > 0:
            average_loss /= 2000
          # The average loss is an estimate of the loss over the last 2000 batches.
          print('Average loss at step ', step, ': ', average_loss)
          average_loss = 0

        if step % 1000 == 0:
          sim = similarity.eval()
          for i in range(validation_size):
            valid_word = reverse_dictionary[valid_examples[i]]
            #valid_word = vocab[valid_examples[i]]
            top_k = 8  # number of nearest neighbors
            nearest = (-sim[i, :]).argsort()[1:top_k + 1]
            log_str = 'Nearest to %s:' % valid_word
            for k in range(top_k):
              close_word = reverse_dictionary[nearest[k]]
              #close_word = vocab[nearest[k]]
              log_str = '%s %s,' % (log_str, close_word)
            print(log_str)
            
        if step % 10000 == 0 and step != 0:  # Save the embeddings every 10000 iterations
            save_embeddings(normalized_embeddings.eval(), normalized = True)
            save_embeddings(embeddings.eval(), normalized = False)
      final_embeddings = normalized_embeddings.eval()
      return final_embeddings


num_steps = 50000
final_embeddings = run(graph, num_steps=num_steps)
# pickle the final  embeddings
with open(path_output+"final_normalized_embeddings.pkl" , 'wb') as file:
    pickle.dump(final_embeddings, file)


def find_similar(word, number_of_words):
    #The cosine similarity
    if word not in dictionary:
        print("word not found")
        return
    index = dictionary[word]
    vector = final_embeddings[index]
    norms = np.sqrt(np.sum(np.square(final_embeddings), 1).reshape(final_embeddings.shape[0], 1))
    norms = final_embeddings / norms
    dot_product = np.matmul(norms, vector)
    sorted_results = (-dot_product).argsort() # This gives indexes instead of sorted elements
     # Now retrieve the top 'number_of_words' words
    for _ in range(1,number_of_words):
         print(reverse_dictionary[sorted_results[_]])

"""
with graph.as_default():
    # Construct the variables for the NCE loss
    nce_weights = tf.Variable(
        tf.truncated_normal([total_words, embedding_size],
                            stddev=1.0 / math.sqrt(embedding_size)))
    nce_biases = tf.Variable(tf.zeros([total_words]))

    nce_loss = tf.reduce_mean(
        tf.nn.nce_loss(weights=nce_weights,
                       biases=nce_biases,
                       labels=train_labels,
                       inputs=embed,
                       num_sampled=num_sampled,
                       num_classes=total_words))

    optimizer = tf.train.GradientDescentOptimizer(1.5).minimize(nce_loss)
    # Add variable initializer.
    init = tf.global_variables_initializer()

num_steps = 50000
nce_start_time = dt.datetime.now()
final_embeddings = run(graph, num_steps)
with open("final_embeddings.pkl" , 'wb') as file:
    pickle.dump(final_embeddings, file)
nce_end_time = dt.datetime.now()
"""
"""
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import word2vec
from nltk.corpus import stopwords
stopwords = stopwords.words("english")


def isnumber(word):
    try:
        int(word)
        return True
    except:
        return False
    
def tsne_plot():
    global stopwords
    "Creates and TSNE model and plots it"
    model = word2vec.word2vec()
    labels = [] 
    tokens = [] 
    count = 0  
    for word in model.dictionary:
        if word not in stopwords and not isnumber(word) :
            tokens.append(model.vectorize(word))
            labels.append(word)
            count+=1
        if count >= 50:
            print("OK", count)
            break
    print(len(tokens))
    
    tsne_model = TSNE(perplexity=40, n_components=2, n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
        
    plt.figure(figsize=(16, 16)) 
    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()
"""
