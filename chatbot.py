import nltk
import string
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer  #TfidfVectorizer:is a tool fromscikit-learn that says "Give me sentences, and I'll convert them into mathematical vectors."
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

questions = []
answers = []

with open("faqs.txt", "r") as file:
    for line in file:
        line = line.strip()

        if line:
            question, answer = line.split("|")

            questions.append(question)
            answers.append(answer)

def preprocess(text):
    text = text.lower()

    tokens = word_tokenize(text)

    tokens = [
    word for word in tokens
    if word not in string.punctuation
    and word not in stop_words
]

    tokens = [
    stemmer.stem(word)
    for word in tokens
]

    return tokens


cleaned_questions = [
    " ".join(preprocess(question))
    for question in questions
]

print(cleaned_questions)

vectorizer = TfidfVectorizer()

faq_vectors = vectorizer.fit_transform(cleaned_questions)

print(faq_vectors.shape)

def get_answer(user_question):   

    cleaned_user_question = " ".join(
        preprocess(user_question)
    )

    user_vector = vectorizer.transform(
        [cleaned_user_question]
    )

    similarities = cosine_similarity(
        user_vector,
        faq_vectors
    )

    print(cleaned_user_question)
    print(similarities)

    best_match_index = np.argmax(similarities)

    best_score = similarities[0][best_match_index]

    if best_score < 0.3:
        return "Sorry, I don't understand that question."
    else:
        return answers[best_match_index]

