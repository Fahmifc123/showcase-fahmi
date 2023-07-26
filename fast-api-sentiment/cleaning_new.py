import pandas as pd
import nltk
from nltk.corpus import stopwords
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

print('Cleaning dimulai...')

# Load stopwords
stop_words = set(stopwords.words('indonesian'))

# Load objects
objects = pd.read_csv('list-objects.csv')

sw_project_name = set()
sw_content = set()
sw_extra_content = set()

for _, row in objects.iterrows():
    project_name = str(row['project_name'])
    content = str(row['content'])
    extra_content = str(row['extra_content'])

    # Tokenize words in the project_name column
    project_name_tokens = project_name.split()
    sw_project_name.update(project_name_tokens)

    # Tokenize words in the content column
    content_tokens = content.split()
    sw_content.update(content_tokens)

    # Replace NaN values with empty string and tokenize words in the extra_content column
    if pd.isnull(extra_content):
        extra_content = ''
    extra_content_tokens = extra_content.split()
    sw_extra_content.update(extra_content_tokens)

# Add additional stop words
stop_words.update(sw_project_name)
stop_words.update(sw_content)
stop_words.update(sw_extra_content)
stop_words.update(['milo', 'jasa marga', 'tol'])

# Initialize stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Function for text cleaning and stemming
def text_cleaning_id(text):
    def casefolding(review):
        review = review.lower()
        return review

    def tokenize(review):
        tokens = nltk.word_tokenize(review)
        return tokens

    def filtering(review):
        review = re.sub(r'@\w+', '', review)
        review = re.sub(r'[^\x00-\x7f]', r'', review)
        review = re.sub(r'(\\u[0-9A-Fa-f]+)', r'', review)
        review = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", review)
        review = re.sub(r'\\u\w\w\w\w', '', review)
        review = re.sub(r'http\S+', '', review)
        review = re.sub(r'@\w+\b', '', review)
        review = re.sub(r'#([^\s]+)', '', review)
        review = re.sub(r"[.,:;+!\-_<^/=?\"'\(\)\d\*]", " ", review)
        return review

    def replaceThreeOrMore(review):
        pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
        return pattern.sub(r"\1", review)

    text = casefolding(text)
    text = filtering(text)
    text = replaceThreeOrMore(text)
    tokens = tokenize(text)
    tokens = [stemmer.stem(token) for token in tokens]
    tokens = [token for token in tokens if token not in stop_words]
    cleaned_review = " ".join(tokens)

    return cleaned_review

df = pd.read_csv('dataset_full_internal.csv')
# Add cleaned_content column to objects DataFrame
df['cleaned_content'] = df['content'].apply(text_cleaning_id)

df.to_csv('data_clean_internal_new.csv', index=False)


print('Cleaning selesai...')