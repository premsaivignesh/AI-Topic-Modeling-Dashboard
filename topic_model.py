from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


def clean_text(text):

    custom_stopwords = set(stopwords.words('english')).union(ENGLISH_STOP_WORDS)

    text = re.sub('[^a-zA-Z]', ' ', text)

    text = text.lower()

    words = text.split()

    words = [
        w for w in words
        if w not in custom_stopwords and len(w) > 2
    ]

    return " ".join(words)


def extract_topics(text_data, num_topics=5):

    cleaned_text = clean_text(text_data)

    documents = re.split(r'[.!?]\s+', cleaned_text)

    documents = [
        doc.strip()
        for doc in documents
        if len(doc.split()) > 5
    ]

    if len(documents) < 2:
        documents = [cleaned_text]

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=1500
    )

    X = vectorizer.fit_transform(documents)

    nmf = NMF(n_components=num_topics, random_state=42)

    nmf.fit(X)

    words = vectorizer.get_feature_names_out()

    topics = []

    for topic in nmf.components_:

        topic_words = [
            words[i]
            for i in topic.argsort()[-6:]
        ]

        topics.append(topic_words)

    topic_strength = nmf.components_.sum(axis=1)

    return topics, topic_strength