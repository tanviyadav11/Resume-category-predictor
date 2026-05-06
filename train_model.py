import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle
import re

print("Loading dataset...")
df = pd.read_csv('UpdatedResumeDataSet.csv')

def cleanResume(txt):
    cleanText = re.sub('http\S+\s', ' ', txt)
    cleanText = re.sub('RT|cc', ' ', cleanText)
    cleanText = re.sub('#\S+\s', ' ', cleanText)
    cleanText = re.sub('@\S+', '  ', cleanText)
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
    cleanText = re.sub('\s+', ' ', cleanText)
    return cleanText

print("Cleaning resumes...")
df['Resume'] = df['Resume'].apply(lambda x: cleanResume(x))

print("Encoding categories...")
le = LabelEncoder()
le.fit(df['Category'])
df['Category'] = le.transform(df['Category'])
pickle.dump(le, open('encoder.pkl', 'wb'))

print("Vectorizing text...")
tfidf = TfidfVectorizer(stop_words='english')
tfidf.fit(df['Resume'])
word_features = tfidf.transform(df['Resume'])
pickle.dump(tfidf, open('tfidf.pkl', 'wb'))

X_train, X_test, y_train, y_test = train_test_split(word_features, df['Category'], random_state=42, test_size=0.2)

print("Training model...")
clf = SVC()
clf.fit(X_train, y_train)

print("Saving model to clf.pkl...")
pickle.dump(clf, open('clf.pkl', 'wb'))

print("Done! You now have clf.pkl, tfidf.pkl, and encoder.pkl ready.")
