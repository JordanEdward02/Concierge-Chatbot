from ctypes.wintypes import tagRECT
from multiprocessing.resource_sharer import stop
from unicodedata import category
from matplotlib.textpath import text_to_path
import nltk
import numpy
import scipy
import os
import pandas as pd
import datetime

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

    def getName(self, text_input):
        """
        Called upon initialising the bot to start by finding the users name. Since this is for use in someones hotel
        for the duration of a stay, this would only occur once and then remember it for the remainded.
        Although for current demo purposes this will be called every time.
        Uses POS tagging and a short list of custom stopwords relating to name-related responses
        """
        # LOADS IN THE NOUNS WHICH ARE LIKELY TO BE SAID IN RESPONSE TO THE NAME PROMPT.
        possessive_nouns = open("dataset/Misc/nameless_nouns.txt", encoding="utf8", errors="ignore", mode="r").read().replace("\n", " ").lower()
        
        # POS TAGS THE INPUT TO IDENTIFY NOUNS. PERSONAL NOUNS ARE THE NAMES WE WANT TO PULL OUT THIS INPUT
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
                self.user = None
                return False

        # FORMATS THE USERS NAME CORRECTLY AND STORES IT WITHIN THE BOT
        self.user = self.user.title()
        return True

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

        #=========
        # Need to add somewhere we store all this transaction data. Also use the input which prompts this in the pipeline to detect information.
        # This would likely only need to look at dates.
        #=========

        booking_name = None
        to_do = ["requests", "name", "date", "email"]
        responses = {}
        #  CODE TO VALIDATE THE BOOKING NAME IF THE SYSTEM KNOWS THE USERS NAME
        if self.user != None:
            if (len(self.user.split(" ")) > 1):
                jprint("Would you like the booking under your name (" + self.user + ")?")
                response = userInput()
                if (confirm(response) == False):
                    jprint("What name would you like to use for your booking?")
                    actual_user = self.user
                    input= userInput()
                    self.getName(input)
                    booking_name = self.user
                    self.user = actual_user
                    if (booking_name == None):
                        booking_name = "_error_"
                    while(len(booking_name.split(" ")) < 2):
                        if (booking_name == "_error_"):
                            jprint("Sorry, but you'll have to be more specific. Please state the first and last name for your booking.")  
                        else:    
                            jprint("Sorry " + self.user + ", but you'll have to be more specific. Please state the first and last name for your booking.")
                        actual_user = self.user
                        input= userInput()
                        self.getName(input)
                        booking_name = self.user
                        self.user = actual_user
                        if (booking_name == None):
                            booking_name = "_error_"
            else:
                jprint("Okay " + self.user + ", what is your full name?")
                input = userInput()
                self.getName(input)
                while(len(self.user.split(" ")) < 2):
                    jprint("Sorry " + self.user + ", but you'll have to be more specific. Please state your first and last name.")
                    input= userInput()
                    self.getName(input)
            if (booking_name != None):
                self.user = booking_name
            to_do.remove("name")

        while len(to_do) > 0:

            # IF NAME IS NOT KNOWN WE PROMPT THE USER TO INPUT TEHEIR NAME. WE USE THE PREVIOUS GETNAME FUNCTION AND REPEAT UNTIL IN A DESIRED FORMAT
            if ("name" in to_do):
                jprint("What name would you like to use for your booking?")
                actual_user = self.user
                input= userInput()
                self.getName(input)
                booking_name = self.user
                self.user = actual_user
                if (booking_name == None):
                    booking_name = "_error_"
                while(len(booking_name.split(" ")) < 2):
                    if (booking_name == "_error_"):
                        jprint("Sorry, but you'll have to be more specific. Please state the first and last name for your booking.")  
                    else:    
                        jprint("Sorry " + booking_name + ", but you'll have to be more specific. Please state the first and last name for your booking.")
                    actual_user = self.user
                    input= userInput()
                    self.getName(input)
                    booking_name = self.user
                    self.user = actual_user
                    if (booking_name == None):
                        booking_name = "_error_"
                if self.user == None:
                    jprint("Is " + booking_name + " your name as well?")
                    input = userInput()
                    if (confirm(input)):
                        self.user = booking_name
                    else:
                        self.getName(input)
                responses["name"] = booking_name
                to_do.remove("name")
            
            # IF THE DATE IS NOT KNOWN WE PROMPT THE USER FOR THE TIMES OF THEIR STAY. THIS HAS TO DEAL WITH DIFFERENT FORMATS
            if ("date" in to_do):
                dates = []
                jprint("When do you want the room?")
                input = userInput()
                days = []
                months = []
                years = []
                with open("./dataset/Transaction/months.txt", encoding="utf8", errors="ignore")as months_file:
                    month_list = months_file.read().lower().split("\n")
                    input_array = input.lower().split(" ")
                    for word_index in range(len(input_array)):
                        # IF WORD IS A MONTH ADD IT TO MONTHS
                        if input_array[word_index] in month_list:
                            months.append(month_list.index(input_array[word_index]) + 1)
                        # IF WORD IS A NUMBER, ADD IT TO DAYS OR YEARS. THIS PRE-PROCESSES TO ENSURE THE STRING IS ONLY DIGITS IF POSSIBLE
                        removed_extra = input_array[word_index].replace("th", "").replace("nd", "").replace("st", "")
                        if removed_extra.isdigit():
                            if (len(removed_extra) == 4):
                                years.append(removed_extra)
                            elif (len(removed_extra) < 3):
                                days.append(removed_extra)
                
                if len(days) == 2 and (len(months) == 1 or len(months) == 2):   

                    if (len(years) == 0):
                        years = [datetime.datetime.now().year,datetime.datetime.now().year]
                    if (len(years) == 1):
                        years.append(years[0])
                    if (len(months) == 1):
                        months.append(months[0])
                    # MAP THE MONTH TO INT
                    input = str(days[0]) + "/" + str(months[0]) + "/" + str(years[0]) + " " + str(days[1]) + "/" + str(months[1]) + "/" + str(years[1])

                if (input.find("/") != -1):
                    date_input = input.split(" ")
                    for item in date_input:
                        # dd/mm/yyyy (or yy) format
                        if ("/" in item):
                            split_dates = item.split("/")
                            try:
                                datetime.datetime(int(split_dates[2]), int(split_dates[1]), int(split_dates[0]))
                                dates.append("/".join(split_dates))
                            except:
                                None

                if (input.find("-") != -1):
                    dates = []
                    date_input = input.split(" ")
                    for item in date_input:
                        # dd/mm/yyyy (or yy) format
                        if ("-" in item):
                            split_dates = item.split("-")
                            try:
                                datetime.datetime(int(split_dates[2]), int(split_dates[1]), int(split_dates[0]))
                                dates.append("/".join(split_dates))
                            except:
                                None
                if (len(dates) == 2):
                    responses["dates"] = dates
                    to_do.remove("date")
                    continue
                else:
                    jprint("Sorry I didn't quite get that, please tell me the start and end dates, with both in the same format (either DD/MM/YYYY or written formally)")
                    continue

            # IF EMAIL IS NOT KNOWN, WE PROMPT THE USER FOR IT. ALSO DO SOME PARSING TO ENSURE IT'S ~VALID BY CHECKING FOR AN @ AND A VALID ENDING
            if ("email" in to_do):
                jprint("We will need a way to contact you and send booking confirmation. What is your email address?")
                email = None
                valid = False
                while valid == False:
                    input = userInput().split(" ")
                    for word in input:
                        if (word.find("@") != -1):
                            words = word.split("@")
                            if (words[1].count(".") == 1):
                                if words[1][-4:] == ".com":
                                    valid = True
                            elif (words[1].count(".") == 2):
                                if words[1][-6:] == ".co.uk" or words[1][-6:] == ".ac.uk":
                                    valid = True
                        if (valid and email == None):
                            email = word
                    if (valid == False):
                        jprint("Sorry I'm not sure that's a valid email. Please send me one like JBot@example.com")
                responses["email"] = email
                to_do.remove("email")

            # IF THE USER HAS A SPECIAL REQUEST, THEY CAN INPUT IT HERE. OTHERWISE, IF THEY SO NO, THEN IT WON'T STORE THE DATA
            if ("requests" in to_do):
                jprint("Do you have any specific requests for your stay?")
                input = userInput()
                if (deny(input) == False):
                    responses["request"] = input.lower()
                to_do.remove("requests")

        if "request" in responses.keys():
            jprint("Great! So I've got a booking for " + responses["name"] + " from " + responses["dates"][0] + " until " + responses["dates"][1] + " with the special request '" + responses["request"] +"'. I also have that you can be contacted via " + responses["email"] + ". Is this correct?")
        else:
            jprint("Great! So I've got a booking for " + responses["name"] + " from " + responses["dates"][0] + " until " + responses["dates"][1] + ". I also have that you can be contacted via " + responses["email"] + ". Is this correct?")
        input = userInput()
        if confirm(input):
            return
        else:
            jprint("Okay lets try this again...")
            self.transaction("")


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
    

def confirm(input):
    input = str(input).split(" ")
    for word in input:
        if ("yes" in word.lower() or "yeah" in word.lower() or word.lower() == "y"):
            return True
    return False

def deny(input):
    input = str(input).split(" ")
    if input[0].lower() == "no" or input[0].lower() == "nope":
        return True
    return False

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
    """
    jprint("Hi! I'm your digital hotel assistant, JBot! What's your name?")
    text_input = input("You: ").lower()
    if (our_bot.getName(text_input)):
        jprint("Nice to meet you " + our_bot.user +". What can I help you with?")
    else:
        jprint("Sorry I didnt manage to get that. Regardless, what can I help you with?")
    """
    while running:
        user_input = userInput()
        intent = our_bot.matchIntent(user_input)
        if intent == QAA:
            our_bot.answerQuestion(user_input)
        elif intent == TRANSACTION:
            our_bot.transaction(user_input)