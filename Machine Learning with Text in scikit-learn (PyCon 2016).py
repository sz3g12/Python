#!/usr/bin/env python
# coding: utf-8

# # Tutorial: Machine Learning with Text in scikit-learn

# # Agenda
#    1. Model building in scikit-learn (refresher)
#    2. Representing text as numerical data
#    3. Reading a text-based dataset into pandas
#    4. Vectorizing our dataset
#    5. Building and evaluating model
#    6. Comparing models
#    7. Examining a model for further insight
#    8. Tuning the vectorizer (discussion)

# ## Part 1: Model building in scikit-learn (refresher)¶

# In[93]:


# load the iris dataset as an example
from sklearn.datasets import load_iris
iris = load_iris()


# In[94]:


# store the features matrix (X) and response vector (y)
X = iris.data
y = iris.target


# **"Features"** are also known as predictors, inputs, or attributes. The **"response"** is also known as the target, label, or output. **"Observations"** are also known as samples, instances, or records.

# In[101]:


# check the shapes of X and y
print(X.shape)
print(y.shape)


# In[102]:


# examine the first 5 rows of the feature matrix (including the feature names)
import pandas as pd
pd.DataFrame(X, columns=iris.feature_names).head()


# In[103]:


# examine the response vector
print(y)


# In order to **build a model**, the features must be **numeric**, and every observation must have the **same features in the same order**.

# In[104]:


# import the class
from sklearn.neighbors import KNeighborsClassifier

# instantiate the model (with the default parameters)
knn = KNeighborsClassifier()

# fit the model with data (occurs in-place)
knn.fit(X, y)


# In order to **make a prediction**, the new observation must have the **same features as the training observations**, both in number and meaning.

# In[122]:


# predict the response for a new observation
knn.predict([[6.3, 3.3, 6. , 2.5]])


# ## Part 2: Representing text as numerical data

# In[2]:


import pandas as pd
import sklearn


# In[3]:


simple_train = ['call you tonight', 'Call me  a cab', 'please call me...PLEASE!']


# In[4]:


# import and instantiate CountVectorizer 
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()
# CounrVectorizer can only recognise 1 dimentional objects


# In[5]:


# learn vocabulary of the traning data
vect.fit(simple_train)


# In[6]:


# examine the fitted vocabulary
vect.get_feature_names()


# In[7]:


# transform training data into a document-term matrix
simple_train_dtm = vect.transform(simple_train)
simple_train_dtm


# In[8]:


# 3*6 document term matrix is 3 rows * 6 columns. 3 refers to 3 documents and 6 refer to 6 terms/features/tokens learnt during fitting.


# In[9]:


# convert sparse matrix to a dense arrary
simple_train_dtm.toarray()


# In[10]:


# show the terms
pd.DataFrame(simple_train_dtm.toarray(), columns = vect.get_feature_names())


# In[11]:


# a corpus of document can be represented with one row per document and one columne 
# per token vectorization is the processingjust now of converting text documents 
# to numerical feature vectors. 
# the term 'bag of words' simply mean that you don't keep track of the order 
# you cannot contruct the original from the document term matrix


# In[12]:


# Check the type of document text matrix
type(simple_train_dtm)


# In[13]:


# examine the sparse matrix contents
print(simple_train_dtm)
# the coordinares indicates the locations of the non zero values)


# In[14]:


# as most documents will typically use a very small subset of the words in the 
# corpus,the matrix will have many feature values that are zeros.
# no. of columns of the matrix is the no. of unique words in the corpus.
# in order to store such a matrix in memory and also speed up the operation,
# sparse representation such as scipy.sparse is used.


# In[15]:


# example text for model testing
simple_test = ["please don't call me"]


# In[16]:


# in order to make a prediction, the new ob needs to have the same 
# features as the training obs, both in number and in meanining.
# thus, we need to use transfer method
simple_test_dtm = vect.transform(simple_test)
simple_test_dtm.toarray()


# In[17]:


# vect.fit(train) learns the vocabulary of the training data
# vect.transform(train) uses the fitted vocabulary to build a document-term matrix from the training data
# vect.transform(test) uses the fitted vocabulary to build a document-term matrix from the testing data and ignore tokens it has
# not seen before (* it still uses fitted vocabulary from train to build the dtms)


# In[18]:


print(vect.transform(simple_test))


# In[19]:


pd.DataFrame(simple_test_dtm.toarray(), columns = vect.get_feature_names())


# ## Part 3: Reading a text-based dataset into pandas

# In[123]:


path = 'C:/Users/zhangxinhua/Desktop/Python/data/sms.tsv'


# In[21]:


sms = pd.read_table(path, header = None, names = ['label', 'message'])


# In[22]:


# convert label to a numerical variable
sms['label'] = sms.label.map({'ham':0, 'spam':1})


# In[23]:


sms.head(10)


# In[24]:


sms.shape


# In[25]:


# examine the class distribution
sms.label.value_counts()


# In[26]:


# how to define X and y for the use of CountVectorizer
X = sms.message
y = sms.label
print(X.shape)
print(y.shape)
# usually X is two dimensional. in this case, it's one dimensional for now but will be transformed with CountVectorizer
# *CountVectorizer can only handle one dimensional object (e.g. cannot even handle (5572, 1) object)


# In[27]:


# split X and y  into training and test sets (*before vectorising them)
# we need to split before vectorizing becuase: 
# 1. the corpus is too large if we don't split first (i.e. train and test togather)
# 2. to simulate the real world, the test set won't know all the featues the training sets have
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 1)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# ## Part 4: Vectorizing our dataset¶

# In[28]:


# instantiate the vectorize
vect = CountVectorizer()


# In[29]:


# learn training data vocabulary, then use it to create 
vect.fit(X_train)
X_train_dtm = vect.transform(X_train)


# In[30]:


# equivalenty: combine fit anad transform into a single step
X_train_dtm = vect.fit_transform(X_train)


# In[31]:


# examine the dtm
X_train_dtm


# In[32]:


X_test_dtm = vect.transform(X_test)
X_test_dtm


# ## Part 5: Building and evaluating a model¶

# In[33]:


# import and instantiate a Multinomial Naive Bayes model
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()


# In[34]:


# train the model using X_training_dtm (timing it with the magic function %time)
# X_train_dtm instead of X_train as the model building requires numbers
get_ipython().run_line_magic('time', 'nb.fit(X_train_dtm, y_train)')


# In[35]:


# make predictions for X_train_dtm
y_pred_class = nb.predict(X_test_dtm)


# In[36]:


# calculate accuracy of class prediction
from sklearn import metrics
metrics.accuracy_score(y_test, y_pred_class)


# In[37]:


# print confusion metrics
metrics.confusion_matrix(y_test, y_pred_class)
# tn,fp 
# fn,tp


# In[38]:


# check the FP and FN to inspect the texts and see how you can possibly improve the model to get them right


# In[39]:


# print message text for the false positves (hams incorrectly predicted as spams)
# meaning y_test = ham and y_pred_class = spam
print(y_test)
print(y_pred_class)
X_test


# In[40]:


# print out messages for false negatives
X_test[(y_pred_class ==  1) & (y_test == 0)]
# y_pred_class and y_test have the same order (the index are preserved)


# In[41]:


# or a more elegant expression since the classes are in numeric
X_test[y_pred_class>y_test]


# In[42]:


# with the same logic, this is the false positives
X_test[y_pred_class<y_test]


# In[43]:


y_test


# In[44]:


# example of false negative
X_test[3132]


# In[45]:


# nb.predict_proba->np arrary that the probability of class being 0 and 1
nb.predict_proba(X_test_dtm)


# In[46]:


# calculate predicted probabilities for X_test_dtm (poorly calibrated->Naive bayes produce extreme values and their
# probabilities should not be interpreted as actual probabilities. i.e.when it says the prob is 1, it's not really 1)
y_pred_prob = nb.predict_proba(X_test_dtm)[:,1]
y_pred_prob


# In[47]:


# y_pred_prob is needed for calculating AUC curve
metrics.roc_auc_score(y_test, y_pred_prob)


# ## Part 6: Comparing models

# In[49]:


# import and instantiate logistic regression from sklearn
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()


# In[50]:


# train the model using X_train_dtm （slower than Naive bayes)
get_ipython().run_line_magic('time', 'logreg.fit(X_train_dtm, y_train)')


# In[51]:


# make class prediction
y_pred_class = logreg.predict(X_test_dtm)


# In[52]:


# calculate predicted probabilities for X_test_dtm (well calibrated)
y_pred_prob = logreg.predict_proba(X_test_dtm)[:,1]
y_pred_prob


# In[53]:


# calculate accuracy 
metrics.accuracy_score(y_test, y_pred_class)


# In[54]:


# calculate AUC
metrics.roc_auc_score(y_test, y_pred_prob)


# ## Part 7: Examining a model for further insight

# In[56]:


# store the vocabulary of X_train 
X_train_tokens = vect.get_feature_names()
len(X_train_tokens)


# In[57]:


# examine the first 50 tokens (from numeric 0 to abc...z to symbols)
print(X_train_tokens[0:50])


# In[58]:


# examine the last 50 tokens
print(X_train_tokens[-50:])


# In[59]:


# Naive bayes counts the number of times each token appears in each class
nb.feature_count_
# the ending '_' after feature_count is a sklearn convention for attributes that are learnt during fitting


# In[60]:


# interpretation f the output above:
# the first token '00' appeared 0 times in ham and 5 times in spam
# the way naive bayes works for text is that it learns the spamminess of each token vs. the cleaness of each token 
# and makes the prediction (for each token, it calculated the conditional probability of that token givne each class
# and it calculate the conditinal probabilities of each class given each token-> A given B, B given A)


# In[61]:


nb.feature_count_.shape 


# In[62]:


# number of times each token appears across all ham messages
ham_token_count = nb.feature_count_[0,:]
ham_token_count


# In[63]:


# number of times each token appears across all spam messages
spam_token_count = nb.feature_count_[1,:]
spam_token_count


# In[64]:


# create a dataframe of tokens with their separate ham and spam counts
tokens = pd.DataFrame({'token': X_train_tokens, 'ham': ham_token_count, 'spam': spam_token_count}).set_index('token')
# pass a dictionary to pd.DataFrame -> Keys are the column names and the values becomes the columns. Set the index as token.
tokens.head()


# In[65]:


len(tokens)
# all the brokendown tokens


# In[66]:


# examine 5 random DataFrame rows (tokens)
tokens.sample(5, random_state = 6)
# 5-> 5 rows 6-> seed
# *the frequencies are the number of times the word/token appeared not the number of messages the token appeared. i.e.some words
# can repeat a few times in the same message
# 'nasty' is still a more spammy word although the frequency is both 1. however, the no. of spammy messages are much fewer


# In[67]:


# Naive bayes counts the number of observations in each class
nb.class_count_


# In[68]:


# calculate the spamminess and hamminess by class
# *add 1 to ham and spam counts to aviod dividing by 0 (also solving the conceptual issue that when seeing 0, e.g. 0 in beloved, 
# we thought the spamminess of this word is 0)
tokens['ham'] = tokens.ham + 1
tokens['spam'] = tokens.spam + 1
tokens.sample(5, random_state = 6)


# In[69]:


# convert the ham and spam counts into frequencies
tokens['ham'] = tokens.ham / nb.class_count_[0]
tokens['spam'] = tokens.spam / nb.class_count_[1]
tokens.sample(5, random_state = 6)


# In[70]:


# calcualte the ratio of spam-to-ham for each token
tokens['spam_ratio'] = tokens.spam / tokens.ham
tokens.sample(5, random_state = 6)
# spam ratio only gives a sense of the level/ranking and shouldn't be interpreted with more ratio/numeric meaning


# In[71]:


# Naive bayes look at the messages as individual words and assess the spamminess of the words with conditional 
# probabilities to predict 


# In[72]:


# examine the DataFrame sorted by spam_ratio
tokens.sort_values('spam_ratio', ascending = False)


# In[73]:


# look up the spam_ratio for a given token
tokens


# In[83]:


# to examine the spam ratio of words
tokens.loc['sms', 'spam_ratio']


# In[84]:


tokens.loc['dating','spam_ratio']


# ## Part 8: Tuning the vectorizer (discussion)

# In[126]:


# show default parameters for CountVectorizer
vect


# **Tuning CountVetorizer:** sklearn try to give the the most sensible default, however, it'e worth tunning, just like models
# some parameters that are useful and relatively easy/effect for tunning
# stop_words: string{'enlish'}, list, or None (default) -> to remove stop words

# In[129]:


# remove English stop words (built-in stop word list)
vect = CountVectorizer(stop_words = 'english')


# **ngram_range:** the lower and upper value of n values. default is (1, 1) meaning all 1-gram -> to identify the important word pairs

# In[130]:


# include 1-gram and 2-grams
vect = CountVectorizer(ngram_range = (1, 2))
# the danger of using 2-gram is that the number of features will grow very quickly and may introduce more noise than signals 
# (e.g. there are a lot of 2-grams only appear once in the dataset)


# **max_df:** max document frequency -> ignore the terms that have a higher frequency than threshold, ranging from 0 to 1. It works similiar as stop words. This is more of corpus specific stop words.

# In[132]:


vect = CountVectorizer(max_df = 0.2)


# **min_df**: min document frequency -> ignore the terms that have a lower frequency than threshold, ranging from 0 to 1 or int. It works similiar as stop words. This is to ignore the terms that have too low frequencies.

# In[ ]:


vect = CountVectorizer(min_df = 2)
# min_df = 2 means that the cut off is two documents


# ## Guidelines for tunning CountVectorizer:
# * Use your knowledge of the problem and the text, and your understanding of the tunning parameters, to help you decide what parameters to tune and how to tune them.
# * Experiment, and let the data tell you the best approaches

# # Summary: small tricks I have learnt
# 1. Green box: edit mode blue box: view mode you can go to view mode by using esc and go to edit mode by using enter
# 2. At view mode, you can use A to add a cell above, B to add a cell below and  X to cut a cell. Use S to save the file
# 3. Use shift + tab to view the parameters of a function/method
# 4. If it's a attribute, there is no (), if it's a function/method, there is ()
# 5. Modelling with sklearn: import->instantiate->fit->predict
# 6. %time to get a rough sense of the time need 
# 7. Naive base is fast. sometimes for big datasets with cross-validation, we can use %time with naive base to test the time needed and then to estimate the time needed for other models. e.g. when logistic regression is too slow, naive bayes only takes 1/4 of the time
# 8. For this test analysis, sklearn don't know what we are analysing. We converted all documents with numeric representation (word count). Thus, any classification model can be used with text problems, just that naive bayes is more popular
# 9. Index is like rows. it can be something like 1, 2, 3, 4..or by names. It can has duplicates too. set_index() can be used to set a column to index. reset_index() can be used to reset index when there are multi-index issue (and when the format goes all weird)
