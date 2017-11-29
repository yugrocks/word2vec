import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import word2vec
from nltk.corpus import stopwords
import pickle


def build_pre_saved_dataset():
    # load pickled dictionary
    with open(r"dictionary.pkl" , 'rb') as f:
        dictionary = pickle.load(f)
    # load embeddings
    with open(r"final_embeddings.pkl" ,'rb') as f:
        embeddings = pickle.load(f)
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return   dictionary, reversed_dictionary, embeddings

    
def collect_pre_saved_data():
    dictionary, reverse_dictionary, embeddings = build_pre_saved_dataset()
    return dictionary, reverse_dictionary, embeddings


stopwords = stopwords.words("english")
# load data
dictionary, reverse_dictionary, embeddings = collect_pre_saved_data()


def tsne_plot(num_words):
    global stopwords
    "Creates and TSNE model and plots it"
    model = word2vec.word2vec()
    labels = []
    tokens = [] 
    count = 0  
    for word in model.dictionary:
        if word not in stopwords :
            tokens.append(model.vectorize(word))
            labels.append(word)
            count+=1
        if count >= num_words: # maximum words to plot
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

# plot
tsne_plot(num_words = 300)
