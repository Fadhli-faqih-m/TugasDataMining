#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


dataset = pd.read_csv("DataJKT.csv")
dataset


# In[3]:


dataset = pd.read_csv('DataJKT.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


# In[4]:


print(x)


# In[5]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=0)


# In[6]:


print(x_train)


# In[7]:


print(x_test)


# In[8]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])


# In[ ]:




