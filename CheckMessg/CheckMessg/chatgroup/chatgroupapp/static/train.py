from sklearn.datasets import fetch_20newsgroups

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import joblib

data = fetch_20newsgroups()
categories = [
        'comp.graphics',
        'rec.sport.baseball',
        'rec.autos',
        'sci.crypt',
        'sci.space',
        
    ]

model = make_pipeline(TfidfVectorizer(), MultinomialNB())

train = fetch_20newsgroups(subset='train', categories=categories)
test = fetch_20newsgroups(subset='test', categories=categories)


model.fit(train.data, train.target)
labels = model.predict(test.data)

#---- เช็คความแม่นยำ

n = len(test.data)
corrects = [ 1 for i in range(n) if test.target[i] == labels[i] ]
print("เช็คความแม่นยำ : ",sum(corrects)*100/n ,"%")

#-- save file.model

filename = 'chatgroup.model'
joblib.dump(model, filename)





#def predict_category(s, train=train, model=model):
    #pred = model.predict([s])
    #return train.target_names[pred[0]]

##pred = model.predict(['my name is tew.'])
##print(train.target_names[pred[0]])

#test.target[0:10]





