import nltk
import numpy as np
import random
import io
import string # for processing standard python strings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

f = io.open('chatbot.txt')
raw = f.read()
raw = raw.lower()

nltk.download('punkt')
nltk.download('wordnet')

# READING IN THE DATA
#Convert to list of sentences
sentence_tokens = nltk.sent_tokenize(raw)
#Convert to list of words
word_tokens = nltk.word_tokenize(raw)

# print(sentence_tokens[:2])
# print(word_tokens[:2])

#PRE-PROCESSING
#Define a function called LemTokens which will return normalized tokens
#Normalized tokens: tokens/words in query are matched despite differences in character sequences of tokens
lemmer = nltk.stem.WordNetLemmatizer()

def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

remove_punct_dict = dict((ord(punct),None) for punct in string.punctuation)

def NormalizeLem(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

# KEYWORD MATCHING
# Greetings
greeting_inputs = ['hey', 'hello', 'hi', 'greetings', "what's up"]
greeting_responses = ['hello', 'hi there', 'hi', 'I am happy you are talking to me', '*acknowledging nod*']

def greeting(sentence):
    for word in sentence.split():
        if word.lower() in greeting_inputs:
            return random.choice(greeting_responses)

# GENERATING RESPONSES
# Concept of document similarity
# Using TF-IDF vectorizer and Cosine similarity
def response(user_response):
    response = ''
    sentence_tokens.append(user_response)

    # Convert collection of raw text documents to matrix of TF-IDF features
    # tokenizer: pass in normalized tokens of real words
    # stop words: remove words that won't be useful in query
    TfidfVec = TfidfVectorizer(tokenizer=NormalizeLem, stop_words='english')
    # Calibrate measurements so data has similar shapes
    tfidf = TfidfVec.fit_transform(sentence_tokens)
    # Computes similarity between samples X and Y
    values = cosine_similarity(tfidf[-1], tfidf)
    index = values.argsort()[0][-2]
    # print('INDEX', index)
    flatten_values = values.flatten()
    # print('FLATTENED VALUES', flatten_values)
    flatten_values.sort()
    req_tfidf = flatten_values[-2]
    # Check to see if user input matches one or more known keywords
    # If doesn't find any input matching keywords: returns response: Can't understand you!
    if(req_tfidf == 0):
        response += "Sorry! Can't understand you!"
        return response
    else:
        response += sentence_tokens[index]
        return response

#Conversation depending on user input
flag=True
print("Chatty: My name is Chatty. I will answer your queries about Chatbots. If you want to exit, type Bye!")
while(flag==True):
    user_response = input("You: ")
    user_response=user_response.lower()
    if(user_response!='bye'):
        if(user_response =='thanks' or user_response =='thank you' ):
            flag=False
            print("Chatty: No problem!")
        else:
            if(greeting(user_response)!=None):
                print("Chatty: "+ greeting(user_response))
            else:
                print("Chatty: ")
                print(response(user_response))
                sentence_tokens.remove(user_response)
    else:
        flag = False
        print("Chatty: See ya! Take care!")
