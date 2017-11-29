import numpy as np
import pickle


class word2vec:
    def __init__(self):
        # Load the dictionary of words and the model(normalized embeddings)
        print("Loading Data...")
        with open("dictionary.pkl",'rb') as file:
            self.dictionary = pickle.load(file)
        self.reversed_dictionary = dict(zip(self.dictionary.values(), self.dictionary.keys()))
        with open("final_normalized_embeddings.pkl" , 'rb') as file:
           self.embeddings = pickle.load(file)
        print("Done")

    def vectorize(self, word):
        return self.embeddings[self.dictionary[word]]

    def get_similar_words(self, word, number_of_words = 10):
        if word not in self.dictionary:
            print("word not found")
            return "Null"
        #The cosine similarity
        vector = self.vectorize(word)
        dot_product = np.matmul(self.embeddings, vector)
        sorted_results = (-dot_product).argsort() # This gives indexes instead of sorted elements
         # Now retrieve the top 'number_of_words' words
        similar_words = []
        for _ in range(1,number_of_words):
             similar_words.append(self.reversed_dictionary[sorted_results[_]])
        return similar_words
             
