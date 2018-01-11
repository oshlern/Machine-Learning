# Following code on http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

# which newsgroups we want to download
newsgroup_names = ['comp.graphics', 'rec.sport.hockey', 'sci.electronics', 'sci.space']

# get the newsgroup data (organized much like the iris data)
newsgroups = fetch_20newsgroups(categories=newsgroup_names, shuffle=True, random_state=265)

# Convert the text into numbers that represent each word (bag of words method)
word_vector = CountVectorizer()
word_vector_counts = word_vector.fit_transform(newsgroups.data)

# Account for the length of the documents:
#   get the frequency with which the word occurs instead of the raw number of times
term_freq_transformer = TfidfTransformer()
term_freq = term_freq_transformer.fit_transform(word_vector_counts)

# Train the Naive Bayes model
model = MultinomialNB().fit(term_freq, newsgroups.target)

# Predict some new fake documents
fake_docs = [
    'That GPU has amazing performance with a lot of shaders',
    'The player had a wicked slap shot',
    'I spent all day yesterday soldering banks of capacitors',
    'Today I have to solder a bank of capacitors',
    'NASA has rovers on Mars',
    'That was a devastating loss',
    'Programmers are weak',
    'Why did you steal my cheese',
    'We won by 5 goals']
fake_counts = word_vector.transform(fake_docs)
fake_term_freq = term_freq_transformer.transform(fake_counts)

predicted = model.predict(fake_term_freq)
print 'Predictions:'
for doc, group in zip(fake_docs, predicted):
    print '\t{0} => {1}'.format(doc, newsgroups.target_names[group])

probabilities = model.predict_proba(fake_term_freq)
print 'Probabilities:'
print probabilities
