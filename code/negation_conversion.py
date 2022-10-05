# -*- coding: utf-8 -*-
# Configured with Python 3.7
import json
from spacy import displacy
from pathlib import Path
import spacy

nlp = spacy.load("en_core_web_sm")
example_fn = "./data/exemples_t1.json"
output_path = Path("dependency_plot.svg")
example_2 = "./data/test1_t1.json"


def load_examples(filename):
    with open(filename, 'r') as fp:
        example_list = json.load(fp)
    return example_list


def tokenizer_cleaner(file):
    word_list = nlp(file)
    token_list = []
    for token in word_list:
        if not token.is_punct and not token.is_stop and not token.is_digit:
            token_list.append(token)
    return token_list


def tokenizer_text_sentence(file):
    doc = nlp(file)
    sentence_list = []
    for sentence in doc.sents:
        sentence_list.append(sentence)
    return sentence_list


def tokenizer_sentence(file):
    doc = nlp(file)
    sentence_list = []
    for sentence in doc.sents:
        sentence_list.append(sentence.text)
    return sentence_list


def tokenizer_word(sentence):
    word_list = nlp(sentence)
    token_list = []
    for token in word_list:
        token_list.append(token)
    return token_list


def print_tree(sentence):
    doc = nlp(sentence)
    svg = displacy.render(doc, style='dep')
    return output_path.open("w", encoding="utf-8").write(svg)


word_portee   = ['but',"as","liked","even","which","and","while","why",'.', '...']
word_negatifs = ["no","No","not","Not","without","never do","Never","Without","none","None","nothing","Nothing","hardly be","hardly","barely","n't"]


def define_portee(tokens):
    '''
    return the first word negatif word of the children list to start apply NOT
    and the last word
    '''
    start, end = tokens[0], tokens[-1]
    for tok in tokens:
        if tok.dep_ == 'neg' or str(tok) in word_negatifs:
            start = tok
        if str(tok) in word_portee:
            end = tok
    return [start, end]

def replace_token(doc, token, children_list):
    '''
    Replace a token if the root is the last word
    and return the list
    '''
    for tok in doc:
        if tok == token:
            break
        if tok == children_list[-1]:
            children_list[-1] = token
            break
    return children_list

def find_portee(doc):
    '''
    Return a list of the first and the last word to apply NOT
    '''
    # Find negation tokens
    negation_tokens = [tok for tok in doc if tok.dep_ == 'neg' or tok.text in word_negatifs]
    
    # Find their head tokens
    negation_head_tokens = [token.head for token in negation_tokens]
    
    negation_portee_start = []
    negation_portee_end = []
    
    
    for token in negation_head_tokens:
        # Find the children tokens
        negation_children = [child for child in token.children]
        if negation_children == []:
            continue
        
        negation_portee = define_portee(negation_children)
        negation_portee = replace_token(doc, token, negation_portee)
        if negation_portee[0].pos_ != 'neg' and negation_portee[0].text not in word_negatifs: continue # No negatif token has been found
        
        negation_portee_start.append(negation_portee[0])
        negation_portee_end.append(negation_portee[-1])
    return negation_portee_start, negation_portee_end

def convert_negated_words(sentence):

    # Preparation for analysis
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(sentence)
    negation_portee_start, negation_portee_end = find_portee(doc)
    i = 0
    result = []
    counter_negated = False
    counter = ''
    for token in doc:

        if counter_negated == True:
            counter = 'NOT_'
        else:
            counter = ''

        if i < len(negation_portee_end) and token == negation_portee_end[i]:
            counter_negated = False
            counter = '' if negation_portee_end[i].text in word_portee else 'NOT_'
            i += 1
        if token.is_punct:
            result.append(token.text)
            continue
        result.append(counter + token.text)
        if i < len(negation_portee_start) and token == negation_portee_start[i]:
            counter_negated = True

    return ' '.join(result)


def negated_sentence(token):
    if token == []:
        return []
    # Cherche la portée négative
    portee1 = []
    portee2 = []
    for word in token:
        portee1.append(word.text)
        if  word.pos_ == 'neg' or word.text in word_negatifs:
            break

    # Cherche la fin de la portée négative
    for word in token[len(portee1):]:
        if word.text in word_portee:
            break
        if word.is_punct:
            portee2.append(word.text)
            continue
        portee2.append("NOT_" + word.text)

    return portee1 + portee2 + negated_sentence(token[len(portee1) + len(portee2):]) 
    

def convert_negated_words_naive(sentence):
    '''
    Faster convertion "naive" to negated words
    '''
    if sentence == '':
        return ''
    token = tokenizer_word(sentence)

    sentences = negated_sentence(token)
    phrase = " ".join(sentences)
    return phrase



def test_examples(filename):
    examples = load_examples(filename)
    for ex in examples:
        sentences = convert_negated_words(ex['S'])
        print("\nPhrase:    {}".format(ex['S']))
        print("Conversion: ", sentences)
        print("real phrase:", ex['N'])
        print(sentences == ex['N'])


if __name__ == '__main__':

    # Test with 2 files
    test_examples(example_fn)
    #test_examples(example_2)

    # output display
    # print(convert_negated_words("I love these albums.  Get all 3 cause they are great"))