"""
CODE REFERENCES
Lab0: https://moodle.nottingham.ac.uk/pluginfile.php/8611970/mod_resource/content/2/COMP3074_Lab0.pdf
Used for some pre-processing, mostly lemmatizer

Lab2: https://moodle.nottingham.ac.uk/pluginfile.php/8612013/mod_resource/content/7/COMP3074_Lab2.pdf
Used for similarity function. This is used in the QA system for cosine similarity and classifier for Intent Matching

Lab3: https://moodle.nottingham.ac.uk/pluginfile.php/8612037/mod_resource/content/5/COMP3074_Lab3.pdf
Used for the QA system and Intent matching

"""

import nltk
import pandas as pd
import datetime

from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

from scipy import spatial

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger")
nltk.download("universal_tagset")
nltk.download('omw-1.4')

lemmatiser = WordNetLemmatizer ()


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
        intent_doc = pd.read_excel("./dataset/Intent_Matching/Intent.xlsx")
        intent_vec = CountVectorizer(lowercase=True, tokenizer=word_tokenize)
        X_train_counts = intent_vec.fit_transform(intent_doc["Prompts"])

        intent_transformer = TfidfTransformer(use_idf=True, sublinear_tf=True).fit(X_train_counts)
        X_train_tf = intent_transformer.transform(X_train_counts)

        clf = OneVsRestClassifier(SVC(probability=True)).fit(X_train_tf, intent_doc["Labels"])

        processed_input = intent_transformer.transform(intent_vec.transform([input]))

        """
        Add in some probability check so if unsure of the intent make it a default no_match
        """
        prediction_values =clf.predict_proba(processed_input)[0]
        if (max(prediction_values) > 0.65):
            return clf.predict(processed_input)
        return UNKNOWN

    #PRIMARILY USING PATTERN MATCHING AND KEYWORD SEARCH
    def transaction(self, input):
        
        booking_name = None
        to_do = ["requests", "name", "date", "email"]
        responses = {}
        
        # SCANS THE INPUT FOR DATES TO CHECK IF THEY ARE ALREADY SPECIFIED WITHIN THE PROMPT WHICH REQUESTS A BOOKING. "I WOULD LIKE TO BOOK A ROOM FROM THE 24TH TO THE 26TH MAY"
        days = []
        months = []
        years = []
        dates = []
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
                        break

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
                        break
        if (len(dates) == 2):
            responses["dates"] = dates
            to_do.remove("date")


        #  CODE TO VALIDATE THE BOOKING NAME IF THE SYSTEM KNOWS THE USERS NAME
        if self.user != None:
            if (len(self.user.split(" ")) > 1):
                jprint("Sure thing! Would you like the booking under your name (" + self.user + ")?")
                response = self.userInput()
                if (confirm(response) == False):
                    jprint("Okay, what name would you like to use for your booking?")
                    actual_user = self.user
                    input= self.userInput()
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
                        input= self.userInput()
                        self.getName(input)
                        booking_name = self.user
                        self.user = actual_user
                        if (booking_name == None):
                            booking_name = "_error_"
                else:
                    booking_name = self.user
            else:
                jprint("Sure thing " + self.user + ", lets get you booked in. What is your full name?")
                input = self.userInput()
                self.getName(input)
                while(len(self.user.split(" ")) < 2):
                    jprint("Sorry " + self.user + ", but you'll have to be more specific. Please state your first and last name.")
                    input= self.userInput()
                    self.getName(input)
            if (booking_name != None):
                self.user = booking_name
            responses["name"] = booking_name
            to_do.remove("name")

        # IF THE DATE IS NOT KNOWN WE PROMPT THE USER FOR THE TIMES OF THEIR STAY. THIS HAS TO DEAL WITH DIFFERENT FORMATS
        while ("date" in to_do):
            broken = False
            dates = []
            jprint("What dates would you like your booking for?")
            input = self.userInput()
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
                            jprint("I'm sorry but that's not a valid date. Please check your request and try again")
                            broken = True
                            break

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
                            jprint("I'm sorry but that's not a valid date. Please check your request and try again")
                            broken = True
                            break
            if (len(dates) == 2):
                responses["dates"] = dates
                to_do.remove("date")
                continue
            elif (broken == False):
                jprint("Sorry I didn't quite get that, please tell me the start and end dates, with both in the same format (either DD/MM/YYYY or written formally)")
                continue

        # IF NAME IS NOT KNOWN WE PROMPT THE USER TO INPUT TEHEIR NAME. WE USE THE PREVIOUS GETNAME FUNCTION AND REPEAT UNTIL IN A DESIRED FORMAT
        while ("name" in to_do):
            jprint("What name would you like to use for your booking?")
            actual_user = self.user
            input= self.userInput()
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
                input= self.userInput()
                self.getName(input)
                booking_name = self.user
                self.user = actual_user
                if (booking_name == None):
                    booking_name = "_error_"
            if self.user == None:
                jprint("Is " + booking_name + " your name as well?")
                input = self.userInput()
                if (confirm(input)):
                    self.user = booking_name
                else:
                    self.getName(input)
            responses["name"] = booking_name
            to_do.remove("name")
        
        # IF EMAIL IS NOT KNOWN, WE PROMPT THE USER FOR IT. ALSO DO SOME PARSING TO ENSURE IT'S ~VALID BY CHECKING FOR AN @ AND A VALID ENDING
        if ("email" in to_do):
            jprint("We will need a way to contact you and send booking confirmation. What is your email address?")
            email = None
            valid = False
            while valid == False:
                input = self.userInput().split(" ")
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
                input = self.userInput()
                if (deny(input) == False):
                    responses["request"] = input.lower()
                to_do.remove("requests")

        if "request" in responses.keys():
            jprint("Great! So I've got a booking for " + responses["name"] + " from " + responses["dates"][0] + " until " + responses["dates"][1] + " with the special request '" + responses["request"] +"'. I also have that you can be contacted via " + responses["email"] + ". Is this correct?")
        else:
            jprint("Great! So I've got a booking for " + responses["name"] + " from " + responses["dates"][0] + " until " + responses["dates"][1] + ". I also have that you can be contacted via " + responses["email"] + ". Is this correct?")
        input = self.userInput()
        if confirm(input):
            jprint("I will send the booking confirmation to you now! Thanks for booking with JBot!")
            jprint("Is there anything else I can help you with?")
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
        count_vec = CountVectorizer(lowercase=True, tokenizer=word_tokenize, stop_words=our_stopwords)
        for q_index in range(len(qaa_doc["Questions"])):
            post = nltk.pos_tag(word_tokenize(qaa_doc["Questions"][q_index]), tagset="universal")
            new_string = []
            for token in post:
                try:
                    new_string.append(lemmatiser.lemmatize(token[0], token[1]))
                except:
                    new_string.append(lemmatiser.lemmatize(token[0]))
            qaa_doc["Questions"][q_index] = " ".join(new_string)
        qaa_matrix = count_vec.fit_transform(qaa_doc["Questions"])
        tf_transformer = TfidfTransformer(use_idf=True).fit(qaa_matrix) # Adding sublinear parameter is not important since all questions are roughly the same size, that is for ratio not flat numbers :)
        qaa_matrix = tf_transformer.transform(qaa_matrix).toarray()

        # PUTS THE INPUT IN THE SAME SPACE AS THE QA MATRIX
        post = nltk.pos_tag(word_tokenize(input), tagset="universal")
        new_string = []
        for token in post:
            try:
                new_string.append(lemmatiser.lemmatize(token[0], token[1]))
            except:
                new_string.append(lemmatiser.lemmatize(token[0]))
        input = " ".join(new_string)
        input_array = count_vec.transform([input])
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

        if similatiry_value < 0.6:
            jprint("Sorry I didn't quite get that. Either rephrase your question or ask a member of staff")
            return

        # OUTPUTS THE ANSWER CORRESPONDING TO THE MOST SIMILAR QUESTION
        jprint(qaa_doc["Answers"][most_similar_question])
        return
    

    def userInput(self):
        if (self.user == None):
            return input("You: ")
        return input(self.user + ": ")

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


def jprint(output):
    print("JBot: " + str(output))
    return

def debug(output):
    print("DEBUG: " + str(output))
    return

if __name__ == "__main__":
    running = True
    our_bot = chatBot()
    jprint("Hi! I'm your digital hotel assistant, JBot! How can I help?")
    while running:
        user_input = our_bot.userInput()
        intent = our_bot.matchIntent(user_input)
        if intent == QAA:
            our_bot.answerQuestion(user_input)
        elif intent == TRANSACTION:
            our_bot.transaction(user_input)
        elif intent == IDENTITY:
            our_bot.getName(user_input)
            jprint("Hi " + our_bot.user + ", it's nice to meet you. Is there anything else I can help you with?")
        elif intent == UNKNOWN:
            jprint("Sorry I dont know what you mean. Could you try rephrasing that?")
        elif intent == STOP:
            jprint("Goodbye!")
            running = False