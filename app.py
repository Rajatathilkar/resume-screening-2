import streamlit as st
import pickle
import re
import nltk
import pandas as pd 
import numpy as np 
nltk.download('punkt')
nltk.download('stopwords')

#loading models
clf = pickle.load(open('clf.pkl','rb'))
tfidfd = pickle.load(open('tfidf.pkl','rb'))
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
import re
import string
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
# Define a function to preprocess text
def preprocess_text(text):
    
    text = text.lower() # lower case conversion
    text = re.sub('[%s]' % re.escape(string.punctuation),'',text) # removal of punctuation
    text = re.sub("\[.*?\]'\w+", '',text) # removal of ticks and next character 
    text = re.sub(r'\w*\d+\w*', '', text) # removal of numbers
    text = re.sub('\n', '', text)# removal of special characters
    text = text.encode('ascii','ignore').decode() # removal of unicode characters
    tokens = word_tokenize(text) # tokenizing the text
    filtered_text = [w for w in tokens if not w in stop_words] # removing stop words
    return " ". join(filtered_text) # returning the process text data.
df=pd.read_csv('UpdatedResumeDataSet.csv')
df
# applying the changes in the dataframe.
df['Resume'] = df['Resume'].apply(preprocess_text)

# Define a function to lemmatize text
lemm = WordNetLemmatizer()
def lemmatize(data):
    text = [lemm.lemmatize(word) for word in data]
    return data

df['Resume'] = df['Resume'].apply(lambda x: lemmatize(x))

from sklearn .feature_extraction.text import TfidfVectorizer 
tfidf=TfidfVectorizer(stop_words='english')
tfidf.fit(df['Resume'])
vectorizer=tfidf.transform(df['Resume'])
def clean_resume(resume_text):
    clean_text = re.sub('http\S+\s*', ' ', resume_text)
    clean_text = re.sub('RT|cc', ' ', clean_text)
    clean_text = re.sub('#\S+', '', clean_text)
    clean_text = re.sub('@\S+', '  ', clean_text)
    clean_text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', clean_text)
    clean_text = re.sub(r'[^\x00-\x7f]', r' ', clean_text)
    clean_text = re.sub('\s+', ' ', clean_text)
    return clean_text
# web app
def main():
    st.title("Resume Screening App")
    uploaded_file = st.file_uploader('Upload Resume', type=['txt','pdf'])

    if uploaded_file is not None:
        try:
            resume_bytes = uploaded_file.read()
            resume_text = resume_bytes.decode('utf-8')
        except UnicodeDecodeError:
            # If UTF-8 decoding fails, try decoding with 'latin-1'
            resume_text = resume_bytes.decode('latin-1')

        cleaned_resume = clean_resume(resume_text)
        input_features = tfidfd.transform([cleaned_resume])
        prediction_id = clf.predict(input_features)[0]
        st.write(prediction_id)

        # Map category ID to category name
        category_mapping = {
            15: "Java Developer",
            23: "Testing",
            8: "DevOps Engineer",
            20: "Python Developer",
            24: "Web Designing",
            12: "HR",
            13: "Hadoop",
            3: "Blockchain",
            10: "ETL Developer",
            18: "Operations Manager",
            6: "Data Science",
            22: "Sales",
            16: "Mechanical Engineer",
            1: "Arts",
            7: "Database",
            11: "Electrical Engineering",
            14: "Health and fitness",
            19: "PMO",
            4: "Business Analyst",
            9: "DotNet Developer",
            2: "Automation Testing",
            17: "Network Security Engineer",
            21: "SAP Developer",
            5: "Civil Engineer",
            0: "Advocate",
        }

        category_name = category_mapping.get(prediction_id, "Unknown")

        st.write("Predicted Category:", category_name)

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

user_question = "Datascience"

## Create a TF-IDF vectorizer to convert the text data and query to a vector representation
tfidf.fit_transform(df['Resume'].values.tolist() + [user_question])

# Get the vector representation of the question and answer
answer_tfidf = tfidf.transform(df['Resume']).toarray()
test_tfidf = tfidf.transform([user_question]).toarray()

# Calculate the cosine similarity between both vectors
cosine_sims = cosine_similarity(answer_tfidf, test_tfidf)

# Get the index of the most similar text to the query
most_similar_idx = np.argmax(cosine_sims)

# Print the most similar text as the answer to the query
print("Resume: ", df.iloc[most_similar_idx]['Resume'])

# python main
if __name__ == "__main__":
     main()

