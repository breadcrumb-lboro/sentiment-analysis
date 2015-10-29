import nltk
from nltk.probability import ELEProbDist
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier

pos_content = [('I love this car', 'positive'),
               ('This view is amazing', 'positive'),
               ('I feel great this morning', 'positive'),
               ('I am so excited about the concert', 'positive'),
               ('He is my best friend', 'positive')]

neg_content = [('I do not like this car', 'negative'),
               ('This view is horrible', 'negative'),
               ('I feel tired this morning', 'negative'),
               ('I am not looking forward to the concert', 'negative'),
               ('He is my enemy', 'negative'),
               ('John is so annoying', 'negative'),
               ]

neut_content = [('He is alright', 'neutral'),
                ('Todays weather is just right', 'neutral'),
                ('He is alright', 'neutral'),
                ('He is alright', 'neutral'),
                ('He is alright', 'neutral'),
                ('He is alright', 'neutral'),
                ]

content = []

for (words, sentiment) in pos_content + neg_content:
    words_filtered = [e.lower() for e in words.split() if len(e) >= 3]
    content.append((words_filtered, sentiment))


def get_words_in_content(content):
    all_words = []
    for (words, sentiment) in content:
        all_words.extend(words)
    return all_words


def get_word_features(wordlist):
    wordlist = nltk.FreqDist(wordlist)
    word_features = wordlist.keys()
    return word_features


def extract_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in document_words)
    return features


word_features = get_word_features(get_words_in_content(content))

tweet = "I really love you"

training_set = nltk.classify.apply_features(extract_features, content)

classifier = nltk.NaiveBayesClassifier.train(training_set)

print classifier.classify(extract_features(tweet.split()))
