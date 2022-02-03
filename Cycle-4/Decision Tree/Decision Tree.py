#!/usr/bin/env python
# coding: utf-8

# In[66]:


import os
import numpy as np
import pandas as pd
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree, metrics, model_selection


# In[67]:


data = pd.read_csv('car.csv',names=['buying','maint','doors','persons','lug_boot','safety','class'])
data.head()


# In[68]:


data.info()


# In[69]:


data['class'],class_names = pd.factorize(data['class'])


# In[70]:


print(class_names)
print(data['class'].unique())


# In[71]:


data['buying'],_ = pd.factorize(data['buying'])
data['maint'],_ = pd.factorize(data['maint'])
data['doors'],_ = pd.factorize(data['doors'])
data['persons'],_ = pd.factorize(data['persons'])
data['lug_boot'],_ = pd.factorize(data['lug_boot'])
data['safety'],_ = pd.factorize(data['safety'])
data.head()


# In[72]:


data.info()


# In[73]:


X = data.iloc[:,:-1]
y = data.iloc[:,-1]


# In[74]:


# split data randomly into 70% training and 30% test
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3, random_state=0)


# In[75]:


# train the decision tree
dtree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
dtree.fit(X_train, y_train)


# In[76]:


# use the model to make predictions with the test data
y_pred = dtree.predict(X_test)
# how did our model perform?
accuracy = metrics.accuracy_score(y_test, y_pred)
print('Accuracy: {:.2f}'.format(accuracy))


# In[77]:


count_misclassified = (y_test != y_pred).sum()
print('Misclassified samples: {}'.format(count_misclassified))


# In[ ]:




