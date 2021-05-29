import nltk
import re
import numpy
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from random import choice


class Bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class ChatBot:
    DOCUMENT_PATH = "switzerland.txt"

    def __init__(self, name):
        self.name = name
        self.document = ''
        self.generate_document()
        self.bot_print(
            "My name is SwissBot. I will answer your questions about Switzerland. If you want to exit, type Bye!", Bcolors.OKGREEN)

    def bot_print(self, text, color, prefix=''):
        if prefix:
            prefix = f"{prefix}: "

        print(f"{color}{prefix}{text}{Bcolors.ENDC}")

    def generate_document(self):
        with open(self.DOCUMENT_PATH) as f:
            lines = f.readlines()

            for line in lines:
                # Remove newline chars
                line = line.replace('\n', ' ')
                # Remove reference numbers
                line = re.sub(r'(\[\w.*\])', '', line)
                # Lowercase line
                line = line.lower()
                # Append to complete text
                self.document += line

    def greeting(self, input):
        greetings = ['Hi', 'Hello', 'Hey', 'Moin']

        if any(i.lower() in input.lower() for i in greetings):
            self.bot_print(choice(greetings), Bcolors.OKGREEN, self.name)
            return True
        return False

    def exit(self):
        self.bot_print("Bye! take care...", Bcolors.OKGREEN, self.name)

    def listen(self):
        while True:
            user_input = input()

            if user_input.lower() == "bye":
                self.exit()
                break

            greeting = self.greeting(user_input)

            if not greeting:
                self.process(user_input)

    def process(self, sentence):
        # converts to list of sentences
        sent_tokens = nltk.sent_tokenize(self.document)
        sent_tokens.append(sentence)

        # TF
        cv = CountVectorizer()
        word_count_vector = cv.fit_transform(sent_tokens)

        # IDF
        tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
        tfidf_transformer.fit(word_count_vector)

        # TF-IDF
        tf_idf_vector = tfidf_transformer.transform(
            word_count_vector)  # computes tfidf as tf*idf

        # Comparing the newly added sentence to the existing sentences
        # compare the first element from the right to the rest of the documents
        vals = cosine_similarity(tf_idf_vector[-1], tf_idf_vector)
        vals = vals.flatten()  # returns a copy of the array collapsed into one dimension

        # skip last one, since it is itself (similarity = 1)
        closest = numpy.amax(vals[:-1])

        # index of the max element
        try:
            closestIndex = int(numpy.where(vals == closest)[0])
            self.bot_print(sent_tokens[closestIndex],
                           Bcolors.OKGREEN, self.name)
        except TypeError:
            self.bot_print("Red Bärndütsch du Mulaff!",
                           Bcolors.FAIL, self.name)


def main():
    bot = ChatBot(name="SwissBot")
    bot.listen()


if __name__ == "__main__":
    main()
