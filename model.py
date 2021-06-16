from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from sklearn.tree import DecisionTreeClassifier
import nltk
import gensim
import nltk as nl
from sklearn.feature_extraction import stop_words
from nltk.stem import PorterStemmer 
import pandas as pd
# nltk.download('punkt')
# nltk.download("stopwords")
import pickle
import numpy as np
import tensorflow as tf
# from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input

def building_photo_modelcnn(uploaded_image):
	imgs = Image.open(uploaded_image)
	model = load_model('model_vgg19.h5')
	# imgs = load_img(image, target_size=(224,224))
	imgs = imgs.resize((224, 224))
	imgs = img_to_array(imgs)
	imgs = np.expand_dims(imgs,axis=0)
	imgs = preprocess_input(imgs)
	return int(np.argmax(model.predict(imgs), axis=1)[0])

def building_photo_model(uploaded_image):
	# imgs = Image.open(imgs)
	imgs = Image.open(uploaded_image)
	model = pickle.load(open('adsclassifymodel.sav','rb'))
	# imgs = load_img(image, target_size=(224,224))
	imgs = imgs.resize((224, 224))
	imgs = img_to_array(imgs)
	imgs = imgs/255
	imgs = np.expand_dims(imgs,axis=0)
	imgs = imgs.reshape(imgs.shape[0],-1)
	return int(model.predict(imgs)[0])
	# return type(imgs)


def building_text_model(statement):
	df1 = pd.read_csv("text_pred.csv")
	df1 = df1.dropna()
	nltk_stopwords = nl.corpus.stopwords.words('english')
	gensim_stopwords = gensim.parsing.preprocessing.STOPWORDS
	sklearn_stopwords = stop_words.ENGLISH_STOP_WORDS
	combined_stopwords = sklearn_stopwords.union(nltk_stopwords,gensim_stopwords)
	porter_stemmer = PorterStemmer()
	statement = statement.lower()
	statement1 = statement.split()
	line = ""
	for i in statement1:
		if i not in combined_stopwords:
			line = line + i + " "
	statement = line
	statement = statement.rstrip()
	statement = porter_stemmer.stem(statement)
	for word in statement:
		if word.isalpha() == False:
			statement = statement.replace(word," ")
	
	X = df1['ad']
	y = df1['Category']
	tfidf_vectorizer = TfidfVectorizer(tokenizer = word_tokenize, max_features = 300)
	tfidf_general = tfidf_vectorizer.fit_transform(X)
	tfidf_features = tfidf_vectorizer.get_feature_names()
	dt = DecisionTreeClassifier()
	dt.fit(tfidf_general, y)

	# df1.loc[len(df1)] = statement
	df2 = {'ad': statement}
	df1 = df1.append(df2, ignore_index = True)

	X = df1['ad']
	tfidf_general = tfidf_vectorizer.fit_transform(X)
	tfidf_features = tfidf_vectorizer.get_feature_names()

	dt_pred = dt.predict(tfidf_general)
	df1['dt_pred'] = dt_pred
	# return df['svc_pred']
	# df1.to_csv("delete abhi.csv")
	if df1['dt_pred'][len(df1)-1] == '0':
		# return statement + ' was classified as a non-ad'
		return False
	elif df1['dt_pred'][len(df1)-1] == '1':
		# return statement + ' was classified as an ad'
		return True
	else:
		return type(df1['dt_pred'][len(df1)-1])

# print(building_text_model("hello yo yo honey singh"))
