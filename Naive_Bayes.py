#!/usr/bin/env python
# coding: utf-8

# In[41]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
dataset = pd.read_csv('iris.csv')
dataset.describe()


# In[46]:


dataset = pd.read_csv('iris.csv')
X = dataset.iloc[:,:4].values
y = dataset['variety'].values
dataset.head(5)


# In[49]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


# In[50]:


from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test) 
y_pred


# In[51]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


# In[52]:


from sklearn.metrics import accuracy_score 
print ("Accuracy : ", accuracy_score(y_test, y_pred))
cm


# In[53]:


df = pd.DataFrame({'Real Values':y_test, 'Predicted Values':y_pred})
df


# In[ ]:




