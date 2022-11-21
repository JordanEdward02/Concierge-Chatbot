from ctypes.wintypes import tagRECT
from multiprocessing.resource_sharer import stop
from unicodedata import category
from matplotlib.textpath import text_to_path
import nltk
import numpy
import scipy
import os
import pandas as pd

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from scipy import spatial

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger")
nltk.download("universal_tagset")


STOP = -1
UNKNOWN = 0
IDENTITY = 1
TRANSACTION = 2
QAA = 3

class chatBot():
    user = None
    intent_matrix = None
    intent_vocab = None
    intent_size = None

    def __init__(self):
        """
        #CREATES ID_CORPUS WHICH IS A LIST OF ALL SENTENCES IN EACH DATASET
        intent_corpus = {} #Corpus of all the words in all the files
        full_vocab = []
        path = "./dataset/Intent_Matching/"
        for file in os.listdir(path):
            file_path = path + os.sep + file
            with open(file_path, encoding="utf8", errors="ignore", mode="r") as doc:
                content = doc.read()
                content = content.replace("\n", " ")
                content_tokenized = list(dict.fromkeys(word_tokenize(content)))
                intent_corpus[file] = content
                full_vocab += [word for word in content_tokenized if word not in full_vocab]

        #Forming the term-dictionary matrix
        self.intent_size = (len(intent_corpus), len(full_vocab))
        doc_matrices = numpy.zeros(shape=self.intent_size, dtype=int)
        for word in full_vocab:
            for doc in intent_corpus:
                count = 0
                for corpus_word in intent_corpus[doc].split():
                    if word == corpus_word:
                        count +=1 
                doc_matrices[list(intent_corpus.keys()).index(doc)][full_vocab.index(word)] = count
        self.intent_matrix = doc_matrices
        self.intent_vocab = full_vocab
        """
        return

    def getName(self):
        """
        Called upon initialising the bot to start by finding the users name. Since this is for use in someones hotel
        for the duration of a stay, this would only occur once and then remember it for the remainded.
        Although for current demo purposes this will be called every time.
        Uses POS tagging and a short list of custom stopwords relating to name-related responses
        """
        # LOADS IN THE NOUNS WHICH ARE LIKELY TO BE SAID IN RESPONSE TO THE NAME PROMPT.
        possessive_nouns = open("dataset/Misc/nameless_nouns.txt", encoding="utf8", errors="ignore", mode="r").read().replace("\n", " ").lower()
        jprint("Hi! I'm your digital hotel assistant, JBot! What's your name?")
        
        # POS TAGS THE INPUT TO IDENTIFY NOUNS. PERSONAL NOUNS ARE THE NAMES WE WANT TO PULL OUT THIS INPUT
        text_input = input("You: ").lower()
        pos_tagged = dict(nltk.pos_tag(word_tokenize(text_input), tagset="universal"))
        nouns = []
        for word, pos in pos_tagged.items():
            if pos == "NOUN":
                nouns.append(word)

        # REMOVES THE NOUNS WHICH ARE KNOWN TO NOT BE NAMES
        possible_names = []
        for item in nouns:
            if item not in possessive_nouns:
                possible_names.append(item)
        if len(possible_names) > 0:
            self.user = " ".join(possible_names)

        # TRIES TO FIND THE WORD 'IS' IF THE NOUN TECHNIQUE FAILS AS A LAST RESORT TO IDENTIFY THE NAME
        if self.user == None:
            try:
                is_location = text_input.split(" ").index("is")
                self.user = str(text_input.split(" ")[is_location+1])
            except:
                jprint("Sorry I didnt manage to get that. Regardless, what can I help you with?")
                return

        # FORMATS THE USERS NAME CORRECTLY AND STORES IT WITHIN THE BOT
        self.user = self.user.title()
        jprint("Nice to meet you " + self.user +". What can I help you with?")
        return

    def matchIntent(self, input):
        """
        #Finds the intent of the string, based of the categories available
        category = UNKNOWN
        formatted_input = input.lower()
        input_array = []
        for intent_word in self.intent_vocab:
            count = 0
            for input_word in formatted_input.split():
                if input_word == intent_word:
                    count +=1
            input_array.append(count)
        most_similar = 0
        similatiry_value = 0.0
        count = 0
        for possible_intent in self.intent_matrix:
            dist = 1 - spatial.distance.cosine(input_array, possible_intent) 
            if (dist) > similatiry_value:
                similatiry_value = dist
                most_similar = count
            count += 1
        file_id = os.listdir("./dataset/categories/")[most_similar]
        if file_id == "identity.txt":
            category = IDENTITY
        elif file_id == "stop.txt":
            category = STOP
        """
        category = TRANSACTION
        return category

    def transaction(self, input):
        to_do = ["name", "date", "room", "email", "phone", "requests"]

        if self.user != None:
            jprint("Would you like the booking under the name " + self.user + "?")
            response = userInput()
            if ("yes" in response.lower() or "yeah" in response.lower()):
                to_do.remove("name")
            else:
                jprint("What name would you like to put it under?")
                name_input = userInput()


        while len(to_do) > 0:
            # DO STUFF HERE
            test = 0
        jprint("I think you want to make a transaction")
        return

    def answerQuestion(self, input):
        """
        A subset of the bot that runs the input through cosine similarity function of a small questions dataset relating to hotel recpeiton questions.
        Also does a few sanity checks, so if none of the input nouns are present it wont output a response as it's almost always going to be wrong.
        In addition it checks if similarity is above a given threshold.
        """
        qaa_doc = pd.read_excel("./dataset/QAA/QAA.xlsx")

        # CUSTOM STOPWORDS ARE LOADED. THESE ARE INPUT INTO THE QA DATASET AND OUR INPUT. IT MUST BE A CUSTOM LIST AS WORDS LIKE WHEN AND WHERE ARE IMPORTANT IN QUESTIONS
        our_stopwords = open("./dataset/Misc/myStopwords.txt", "r").read().split("\n")

        # CREATES THE QA MATRIX
        count_vec = CountVectorizer(tokenizer=word_tokenize, stop_words=our_stopwords)
        qaa_matrix = count_vec.fit_transform(qaa_doc["Questions"])
        tf_transformer = TfidfTransformer(use_idf=True).fit(qaa_matrix) # Adding sublinear parameter is not important since all questions are roughly the same size, that is for ratio not flat numbers :)
        qaa_matrix = tf_transformer.transform(qaa_matrix).toarray()

        # PUTS THE INPUT IN THE SAME SPACE AS THE QA MATRIX
        input_vec = CountVectorizer(tokenizer=word_tokenize, vocabulary=count_vec.vocabulary_, stop_words=our_stopwords) # use the same vocabulary as qa dataset so we it is in the same dimensions/space
        input_array = input_vec.transform([input])
        input_array = tf_transformer.transform(input_array).toarray()


        # IF ALL OF THE VALUES IN THE LIST ARE 0, SO IF NONE OF THE PROCESSED WORDS IN THE INPUT ARE IN THE QA VOCAB, WE STOP 
        if all(item == 0 for item in input_array[0]):
            jprint("Sorry I didn't quite get that. Either rephrase your question or ask a member of staff")
            return

        # CALCULATE THE SIMILARITY OF THE THE INPUT COMPARED WITH OUR QUESTIONS IN THE DATASET
        most_similar_question = 0
        similatiry_value = 0.0
        count = 0
        for question in qaa_matrix:
            dist = 1 - spatial.distance.cosine(input_array, question) 
            if (dist) > similatiry_value:
                similatiry_value = dist
                most_similar_question = count
            count += 1

        # ASSUMES THE QUESTION IS NOT CORRECT IF THE SIMILARITY IS TOO LOW
        debug(similatiry_value) 
        if similatiry_value < 0.65:
            jprint("Sorry I didn't quite get that. Either rephrase your question or ask a member of staff")
            return

        # OUTPUTS THE ANSWER CORRESPONDING TO THE MOST SIMILAR QUESTION
        jprint(qaa_doc["Answers"][most_similar_question])
        return
    
    def disambiguate(self):
        jprint("Did you mean something like.... ?")
        return 

def userInput():
    return input("You: ")

def jprint(output):
    print("JBot: " + str(output))
    return

def debug(output):
    print("DEBUG: " + str(output))
    return

if __name__ == "__main__":
    running = True
    our_bot = chatBot()
    our_bot.getName()
    while running:
        user_input = userInput()
        intent = our_bot.matchIntent(user_input)
        if intent == QAA:
            our_bot.answerQuestion(user_input)
        elif intent == TRANSACTION:
            our_bot.transaction(user_input)