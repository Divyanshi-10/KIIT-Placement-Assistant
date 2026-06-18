import nltk
import string
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer  #TfidfVectorizer:is a tool fromscikit-learn that says "Give me sentences, and I'll convert them into mathematical vectors."
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from nltk.stem import PorterStemmer

nltk.download('punkt')
nltk.download('punkt_tab')
stemmer = PorterStemmer()

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

while True:

    user_question = input("You: ")

    if user_question.lower() == "exit":
        print("Bot: Goodbye!")
        break

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
    print(similarities)

    best_match_index = np.argmax(similarities)

    best_score = similarities[0][best_match_index]

    if best_score < 0.3:
        print("Bot: Sorry, I don't understand that question.")
    else:
        print("Bot:", answers[best_match_index])