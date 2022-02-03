#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[4]:


dataset = pd.read_csv("cancer.csv")
dataset.head()
dataset.info()
X = dataset.iloc[:, 2:35].values
print(X)
y = dataset.iloc[:, 1].values
print(y)


# In[5]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)


# In[6]:


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)


# In[7]:


from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, y_pred))


# In[8]:


from sklearn.metrics import accuracy_score
print ("Accuracy : ", accuracy_score(y_test, y_pred))
df = pd.DataFrame({'Real Values':y_test, 'Predicted Values':y_pred})


# In[ ]:




