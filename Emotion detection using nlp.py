#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import os 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
import seaborn as sns

#import warnings
#warnings.filterwarnings('ignore')
from wordcloud import WordCloud


# In[9]:


df = pd.read_csv("text.csv")


# In[10]:


df.head()


# In[11]:


df.info()


# In[12]:


df.drop(columns="Unnamed: 0", inplace=True)
df.head()


# In[13]:


df.info()


#  Label description
# - Sadness 0 
# - Joy 1
# - Love 2
# - Anger 3
# - Fear 4
# - surprise 5
# 

# In[14]:


# Check and remove duplicate records
df.duplicated().sum()


# In[15]:


df_dup = df[df.duplicated()]

df_dup


# ## EDA 

# In[16]:


df.info()


# In[17]:


emotion_map = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "suppride"
}
df["label"] = df["label"].map(emotion_map)


# In[18]:


df["label"].value_counts()


# ## Visualize the frequency of each label

# In[19]:


count_labels = df["label"].value_counts()

fig, axs = plt.subplots(1, 2, figsize=(12, 6))

axs[0].pie(count_labels, labels=count_labels.index, autopct='%1.1f%%', startangle=150, colors=sns.color_palette("Blues"))
axs[0].set_title('Frequency distribution of labels', fontsize=15, fontweight='bold')

sns.barplot(x=count_labels.index, y=count_labels.values, ax=axs[1], palette="Blues")
axs[1].set_title("Frequency of appearance of each label", fontsize=14)
axs[1].bar_label(axs[1].containers[0])

plt.tight_layout()
plt.show()


# - Create a separate set of texts to represent the word space within the label word

# In[20]:


df_sadness = df[df["label"]=="sadness"]
df_joy = df[df["label"]=="joy"]
df_love = df[df["label"]=="love"]
df_anger = df[df["label"]=="anger"]
df_fear = df[df["label"]=="fear"]
df_suppride = df[df["label"]=="suppride"]


# - Combines all the text in the Text column into a single string, separated by spaces

# In[21]:


combine_sadness = " ".join(df_sadness['text'])
combine_joy = " ".join(df_joy['text'])
combine_love = " ".join(df_love['text'])
combine_anger = " ".join(df_anger['text'])
combine_fear = " ".join(df_fear['text'])
combine_surprise = " ".join(df_suppride['text'])


# ## Create and visualize word clouds on each labels

# In[22]:


sadness_wordcloud = WordCloud(width=1500, height=1000, background_color='white', colormap='Blues_r').generate(combine_sadness)
joy_wordcloud = WordCloud(width=1500, height=1000, background_color='white', colormap='PiYG_r').generate(combine_joy)
love_wordcloud = WordCloud(width=1500, height=1000, background_color='white', colormap='Purples_r').generate(combine_love)
anger_wordcloud = WordCloud(width=1500, height=1000, background_color='white', colormap='Greys').generate(combine_anger)
fear_wordcloud = WordCloud(width=1500, height=1000, background_color='white', colormap='mako_r').generate(combine_fear)
surprise_wordcloud = WordCloud(width=1500, height=1000, background_color='white', colormap='Greens').generate(combine_surprise)


# In[23]:


plt.figure(figsize=(25, 25)) 

plt.subplot(3, 2, 1)  # Bố trí subplot (Số hàng, Số cột, vị trí)
plt.imshow(sadness_wordcloud, interpolation='bilinear')
plt.title('Sadness Text')
plt.axis('off')  

plt.subplot(3, 2, 2)
plt.imshow(joy_wordcloud, interpolation='bilinear')
plt.title('Joy Text')
plt.axis('off')

plt.subplot(3, 2, 3)  
plt.imshow(love_wordcloud, interpolation='bilinear')
plt.title('Love Text')
plt.axis('off')

plt.subplot(3, 2, 4)
plt.imshow(anger_wordcloud, interpolation='bilinear')
plt.title('Anger Text')
plt.axis('off')

plt.subplot(3, 2, 5)  
plt.imshow(fear_wordcloud, interpolation='bilinear')
plt.title('Fear Text')
plt.axis('off')

plt.subplot(3, 2, 6) 
plt.imshow(surprise_wordcloud, interpolation='bilinear')
plt.title('Suppride Text')
plt.axis('off')

plt.show()


# ## Check the sentence length distribution

# In[24]:


df['sentence_length'] = df['text'].apply(lambda x: len(x.split()))
counts, bins = np.histogram(df['sentence_length'], bins=30)
plt.figure(figsize=(10, 6))

sns.histplot(df['sentence_length'], bins=30, kde=True, color="royalblue", edgecolor="black")
for i in range(len(counts)):
    # Màu sắc theo tần suất
    plt.bar(bins[i], counts[i], width=bins[i+1]-bins[i], color=plt.cm.Blues(counts[i]/max(counts)), edgecolor="black")

plt.title("Sentence length distribution", fontsize=14, fontweight='bold')
plt.xlabel("Number of words", fontsize=12)
plt.ylabel("Quantity", fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# ## Find the shortest and longest sentences

# In[25]:


df['sentence_length'] = df['text'].apply(lambda x: len(x.split()))
long = df.loc[df['sentence_length'].idxmax()]

print("Longest Sentences is:")
print(f"{long['text']}\t--> Label: [{long['label']}]")
print(f"Sentence length: [{long['sentence_length']}] word")


# In[26]:


short = df.loc[df['sentence_length'].idxmin()]
print("Shortest Sentence is: ")
print(f"{short['text']}\t--> Label: [{short['label']}]")
print(f"Sentence length: [{short['sentence_length']}] words")


# ## Data cleaning

# In[27]:


# Process and represent text data
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter # Count words or characters for frequency analysis
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer



# In[28]:


def preprocess_text(text):
    text = text.lower()  # Lowercase the text
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'\b\w{1,2}\b', '', text)  # Remove short words
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers

    stop_words = set(stopwords.words('english')) - {'not', 'no', 'nor'}  # Remove stopwords except negations
    lemmatizer = WordNetLemmatizer()

    # Tokenize text to handle negations like isn't, wasn't, aren't, etc.
    tokens = nltk.word_tokenize(text)
    tokens_lemmatized = []

    for token in tokens:
        if token in stop_words:
            continue
        if token in ['not', 'no', 'nor']:
            tokens_lemmatized.append(token)
        elif token.endswith("n't"):
            tokens_lemmatized.append(token)  # Keep contractions like "isn't"
        else:
            tokens_lemmatized.append(lemmatizer.lemmatize(token))

    text = ' '.join(tokens_lemmatized)

    return text


# In[29]:


df['preprocessed_text'] = df['text'].apply(preprocess_text)


# In[30]:


df.head()


# ## Model Training

# In[31]:


# Machine learning library
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.linear_model import SGDClassifier


# In[32]:


emotion_map = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise"
}

label_encoder = LabelEncoder()
df['label_en']= label_encoder.fit_transform(df['label'])
df['label'] = df['label_en'].map(emotion_map)
y=df['label'].values


# In[33]:


y=df['label'].values


# In[34]:


tfidf = TfidfVectorizer(max_features=30000)
X = tfidf.fit_transform(df["preprocessed_text"])


# In[35]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train.shape
X_test.shape


# In[36]:


def evolution_models(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("------------------------------------------")

    labels = df['label'].unique()
    labels = list(labels)

    # Caculater to Confution Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title("Confution matrix heatmap:")
    plt.xlabel("Actual labels")
    plt.ylabel("Predicted labels")
    plt.show()


# ## Model 1: LinearSVC 

# In[37]:


def svc_base_model():
    model = LinearSVC()
    model.fit(X_train, y_train)
    return model
svc_base = svc_base_model()
evolution_models(svc_base, X_test, y_test)


# ## Model 2: Naive Bayes Classifier 

# In[38]:


def NBmultil():
    model = MultinomialNB(alpha=0,force_alpha=False, fit_prior=False)
    model.fit(X_train, y_train)
    return model
NB_base = NBmultil()
evolution_models(NB_base, X_test, y_test)


# ## Model 3: Logistic Regression with SGD 

# In[39]:


def LogisticRegressor():
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model

LR_SGD = LogisticRegressor()
evolution_models(LR_SGD, X_test, y_test)


# In[40]:


import pickle
pickle.dump(tfidf,open('vectorizer1.pkl','wb'))
pickle.dump(LR_SGD,open('model1.pkl','wb'))


# In[ ]:





# In[ ]:





# In[41]:


#input_text = input("enter tex") #input("Enter the text for emotion prediction: ")


# In[50]:


import joblib
from sklearn.feature_extraction.text import TfidfVectorizer


model = joblib.load('model1.pkl')
vectorizer = joblib.load('vectorizer1.pkl')


input_text = input("Enter the text for emotion prediction: ")


input_vectorized = vectorizer.transform([input_text])

predicted_emotion = model.predict(input_vectorized)





print(f"Predicted Emotion: {predicted_emotion[0]}")




# In[43]:


df.sample(15)


# In[ ]:





# In[ ]:




