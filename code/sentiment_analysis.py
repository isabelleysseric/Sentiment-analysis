# -*- coding: utf-8 -*-
import json
import numpy as np
import random
import math
from nltk.corpus import sentiwordnet
from negation_conversion import tokenizer_sentence, tokenizer_word, convert_negated_words, convert_negated_words_naive
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
#sentiwordnet.senti_synsets('slow')


reviews_dataset = {
                'train_pos_fn': "./data/train_positive.txt",
                'train_neg_fn': "./data/train_negative.txt",
                'test_pos_fn':  "./data/test_positive.txt",
                'test_neg_fn':  "./data/test_negative.txt"
            }

model_dict = {
                'NB': MultinomialNB(alpha=1.0e-10, fit_prior=True),
                'LG': LogisticRegression(C=0.01, max_iter=200),
                'NN': MLPClassifier(hidden_layer_sizes=100, activation='relu')
            }
          
          

###  CLASS FOR EXTRACTION    
   
class Dataset:

    def __init__(self, dataset, features, naive=False):
        self.features = features
        self.dataset = dataset
        self.naive = naive
        self.vectorizer = CountVectorizer(analyzer='word')
        self.x = {'train' : [], 'test': []}
        self.y = {'train' : [], 'test': []}
        self.result = {'train' : [], 'test': []}
        self.func = {
                        'jurafsky' : self.__jurafsky,
                        'ohana' : self.__ohana, 
                        'combined' : self.__combined, 
                        'words' : self.__words, 
                        'negated_words' : self.__negated_words
                    }


    ### MÉTHODS FOR DATASET
    
    def prepare_data(self):
        self.__prepare_dataset()
        self.__prepare_vectorizer()
        for key in self.dataset:
            data = self.dataset[key]
            index = 'train' if 'train' in key else 'test'
            label = 1 if 'pos' in key else 0
            for review in data:
                self.result[index].append((self.func[self.features](review), label))
        self.__fillup('train')
        self.__fillup('test')
        return self.x["train"], self.y["train"], self.x["test"], self.y["test"]

    def __prepare_dataset(self):
        # Read the dataset and put it into dataset variable
        print("Prepare data")
        for key in self.dataset:
            path = self.dataset[key]
            if 'negated' in self.features:
                path = negated_path(path, self.naive)
            self.dataset[key] = load_reviews(path)

    def __prepare_vectorizer(self):
        if 'words' in self.features:
            dataset_pos = self.dataset['train_pos_fn']
            dataset_neg = self.dataset['train_neg_fn']
            dataset = dataset_pos + dataset_neg
            self.vectorizer.fit(dataset)
        print("Vectorizer ready")

    def __fillup(self, index):
        random.shuffle(self.result[index])
        for x, y in self.result[index]:
            self.x[index].append(x)
            self.y[index].append(y)


    ### MÉTHODS FOR EXTRACTION
    
    def __jurafsky(self, doc):

        # preparation for extraction
        result = np.zeros((6), dtype=float)
        sentences = tokenizer_word(doc)
        text_words, neg_word, pper, exclam_marks, ps, pn = ( [] for i in range(6) )
        pos_score, neg_score = np.zeros((2), dtype=float)
        pronoms = ['i', 'you', 'yourself', 'we', 'me', 'mine', 'myself', 'us', 'your', 'yours', 'our', 'ours', 'ourselves', 'yourselves', 'this', 'these', 'those' ]

        # extraction of attributes
        for word in sentences:
            text_words += [word]                    # number of words in the text
            if word.dep_ == 'neg':                  # negative words
                neg_word += [word]
            if word.text in pronoms:                # 1st and 2nd person pronouns
                pper += [word]
            if str(word) == "!":                    # exclamation marks
                exclam_marks += [word]
            if not word.is_punct and not word.is_stop and not word.is_digit:
                try:
                    w = sentiwordnet.senti_synsets(str(word))
                    w0 = list(w)[0]
                    if w0.pos_score() > 0:          # positive polarity words
                        ps += [word]
                        pos_score += w0.pos_score()
                    if w0.neg_score() > 0:          # negative polarity words
                        pn += [word]
                        neg_score += w0.neg_score()
                except:
                    pass

        # method attributes
        result[0] = len(ps)                         # number of positive polarity words
        result[1] = len(pn)                         # number of negative polarity words
        result[2] = len(neg_word)                   # présence de mots de négation
        result[3] = len(pper)                       # 1st and 2nd person pronoun count
        result[4] = len(exclam_marks)               # number of exclamation mark
        result[5] = math.log(len(text_words))       # text length

        # output display
        # print("nombre  polarité positif: {}\nnombre  polarité negatif: {}\nmots de négation: {}\npronoms personnels: {}\npoint d'exclamation: {}\nNombre de mots: {}\n\n"
        #     .format(result[0], result[1], result[2], result[3], result[4], result[5]))

        return result

    def __ohana(self, doc):

        # preparation for extraction
        result = np.zeros((12), dtype=float)
        text = tokenizer_sentence(doc)
        word_list = tokenizer_word(doc)
        nn, pn, aj, vb, av, ij, sw, ps, pn = ( [] for i in range(9) )
        pos_score, neg_score = np.zeros((2), dtype=float)

        # extraction of attributes
        for word in word_list:
            if word.pos_ == 'NOUN':                   # nouns
                nn += [word]
            if word.pos_ == 'PROPN':                  # proper noun
                pn += [word]
            if word.pos_ == 'ADJ':                    # adjectives
                aj += [word]
            if word.pos_ == 'VERB':                   # verbs
                vb += [word]
            if word.pos_ == 'ADV':                    # adverbs
                av += [word]
            if word.pos_ == 'INTJ':                   # interjections
                ij += [word]
            if word.text in STOP_WORDS:               # stop words
                sw +=  [word]
            if not word.is_punct and not word.is_stop and not word.pos_ == 'NUM':
                try:
                    w = sentiwordnet.senti_synsets(str(word))
                    w0 = list(w)[0]
                    if w0.pos_score() > 0:              # positive word
                        ps += [word]
                        pos_score += w0.pos_score()
                    if w0.neg_score() > 0:              # negative word
                        pn += [word]
                        neg_score += w0.neg_score()
                except:
                    pass

        # method attributes
        result[0] = len(nn)                             # number of nouns
        result[1] = len(pn)                             # number of proper noun
        result[2] = len(aj)                             # number of adjectives
        result[3] = len(vb)                             # number of verbs
        result[4] = len(av)                             # number of adverbs
        result[5] = len(ij)                             # number of interjections
        result[6] = len(text)                           # number of sentences
        result[7] = len(word_list)/len(text)            # average sentence length
        result[8] = len(sw)                             # number of stop words

        if ps == [] or pos_score == 0.0:                # cumulative positive word score
            result[9] = 0.0
        else:
            result[9]= pos_score/float(len(ps))

        if pn == [] or neg_score == 0.0:                # cumulative negative word score
            result[10] = 1.0
        else:
            result[10]= neg_score/float(len(pn))

        if (result[9] or result[10]) != 0.0:            # ratio between positive and negative score
            result[11] = float(result[9] / result[10])
        else:
            result[11] = result[9]

        # output display
        print("Number sentences: {}\nlongueur moyenne phrases: {}\nnouns: {}\nproper noun: {}\nadjectives: {}\nverbs: {}\nadverbs: {}\ninterjections: {}\nmots outils: {}\nScore positif: {}\nScore negatif: {}\nRatio pos/neg: {}\n\n"
              .format(result[6],result[7],result[0],result[1],result[2],result[3],result[4],result[5],result[8],result[9],result[10],result[11] ))

        return result

    def __combined(self, doc):
        return np.concatenate((self.__jurafsky(doc), self.__ohana(doc)))

    def __words(self, doc):

        # document-term matrix
        x = self.vectorizer.transform([doc])

        # method attributs
        x = x.toarray()
        result = []
        for nb in x[0]:
            result.append(nb)

        # output display
        # print("Attributes: {}\nFrequences: {}\n\n".format(self.vectorizer.get_feature_names(), x))

        return np.array(result)

    def __negated_words(self, doc):
        return self.__words(doc)



### MÉTHOD FOR TRAINING AND TESTING METHODES

def train_and_test_classifier(dataset, model, features, naive=False):
    """
    :param dataset: les 4 fichiers utilisées pour entraîner et tester les classificateurs. Voir reviews_dataset.
    :param model: le type de classificateur. NB = Naive Bayes, LG = Régression logistique, NN = réseau de neurones
    :param features: le type d'attributs (features) que votre programme doit construire
                 - 'jurafsky': les 6 attributs proposés dans le livre de Jurafsky et Martin.
                 - 'ohana': les 12 attributs représentant le style de rédaction (Ohana et al.)
                 - 'combined': tous les attributs 'jurafsky' et 'ohaha'
                 - 'words': des vecteurs de mots
                 - 'negated_words': des vecteur de mots avec conversion des mots dans la portée d'une négation
    :param naive: Le type de fonction de negation utilisé
    :return: un dictionnaire contenant 3 valeurs:
                 - l'accuracy à l'entraînement (validation croisée)
                 - l'accuracy sur le jeu de test
                 - la matrice de confusion obtenu de scikit-learn
    """
    print("train and test ...")

    # classification
    dataset_manager = Dataset(dataset, features, naive)
    x_train, y_train, x_test, y_test = dataset_manager.prepare_data()
    clf = model_dict[model]
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    # measure of result
    results = dict()
    results['accuracy_train'] = clf.score(x_train, y_train)
    results['accuracy_test']  = clf.score(x_test, y_test)
    results['confusion_matrix'] = confusion_matrix(y_test, y_pred)

    return results



### MÉTHODS FOR EXTRACTION PROCESSED SENTENCES

def write_file(filename, data, naive):
    path = negated_path(filename, naive)
    f = open(path, "w+")
    json_object = json.dumps(data)
    f.write(json_object)
    f.close()


def write_negated_dataset(dataset, naive=False):
    for key in dataset:
        data = load_reviews(dataset[key])
        tmp = []
        # Transform the dataset to negated if feature is negated wordsfilename
        for review in data:
            tmp.append(convert_negated_words(review)) if not naive else tmp.append(convert_negated_words_naive(review))
        write_file(dataset[key], tmp, naive)


def negated_path(path, naive=False):
    counter_naive = "" if not naive else "naive_"
    path = path.split('/')
    path[-1] = 'negated_' + counter_naive + path[-1]
    path = '/'.join(path)
    return path
    
    
def load_reviews(filename):
    with open(filename, 'r') as fp:
        reviews_list = json.load(fp)
    return reviews_list



### MAIN  

if __name__ == '__main__':
    
    # Write the processed sentences
    naive = True
    #write_negated_dataset(reviews_dataset, naive) # Use this function only for the first launch
    
    # Test the methods
    model = 'NB'
    features = 'ohana'
    results = train_and_test_classifier(reviews_dataset, model=model, features=features, naive=False)
    print("model='{}, features='{}'\nAccuracy - Entraînement: {}\nAccuracy - Test: {}\nMatrice de confusion: {}\n"
          .format(model, features, results['accuracy_train'], results['accuracy_test'], results['confusion_matrix']))
