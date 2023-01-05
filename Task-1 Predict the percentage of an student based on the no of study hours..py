#!/usr/bin/env python
# coding: utf-8

# Linear Regression with python Scikit Learn- Prediction using Supervised ML
# 

# Task 1 : Predict the percentage of an student based on the no. of study hours

# In this regression task we will predict the percentage of marks that a student is expected to score based upon the number of hours they studied. This is simple linear regression task as it involves just two variables.

# ## Name : Subhashree Roy ##

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style


# In[2]:


Data = "http://bit.ly/w-data"
df = pd.read_csv(Data)
df.head()


# In[3]:


df.tail()


# In[4]:


df.shape


# In[4]:


df.size


# In[5]:


#checking the missing value:
df.isnull().sum()


# In[6]:


df.describe()


# In[7]:


df.info()


# # Data Visualization #

# In[9]:


df.plot(kind = 'line')
plt.title("Hours vs Predicted scores ")
plt.xlabel("Hours studied")
plt.ylabel("Percentage Scores")
plt.show()


# In[10]:


style.use('ggplot')
df.plot(kind='line')
plt.title("Hours vs percentage Scores")
plt.xlabel("Hours studied")
plt.ylabel("Percentage Scores")
plt.show()


# In[11]:


df.plot(kind= 'bar')
plt.title("Hours vs percentage Scores")
plt.xlabel("Hours studied")
plt.ylabel("Percentage Scores")
plt.show()


# In[12]:


df.plot(kind= 'scatter' , x='Hours', y='Scores', c='b')
plt.title("Hours vs percentage Scores")
plt.xlabel("Hours studied")
plt.ylabel("Percentage Scores")
plt.show()


# Using Scatter plot we can clearly see there's an linear relation between studied hours and percentage scores.

# # Splitting the Data #

# In[13]:


x=df.iloc[:,:-1].values
y=df.iloc[:,1].values


# In[14]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2, random_state =42) 

print("shape of x train" , x_train.shape)
print("shape of y test" , y_train.shape)
print("shape of y train" , x_test.shape)
print("shape of y test" , y_test.shape)


# # Linear Regression - ML Model Training 

# In[15]:


from sklearn.linear_model import LinearRegression


# In[16]:


lr=LinearRegression()
lr.fit(x_train,y_train)


# In[17]:


m=lr.coef_
m


# In[18]:


c=lr.intercept_
c


# # Testing the model :

# In[19]:


y_pred=lr.predict(x_test)


# In[20]:


#comparison in between actual and predicted
df2=pd.DataFrame({'Actual':y_test,'Predicted':y_pred})
df2


# In[21]:


#visualizing the regression:
df.plot(kind= 'scatter' , x='Hours', y='Scores', label='data')
plt.plot(x_train,m*x_train+c,c='b',label='Line of best fit')
plt.xlabel("Hours studied")
plt.ylabel("Percentage Scores")
plt.legend()
plt.show()


# # Model evaluation

# In[ ]:


#using metrics to find the mean absolute error & r2 to see the accuracy
from sklearn import metrics
from sklearn.metrics import r2_score
print("Accuracy: %.2f" %r2_score(y_test,y_pred))
print('Mean Absolute Error : ', format(metrics.mean_absolute_error(y_test,y_pred)))


# # Predicting the Score with User Input

# In[26]:


hours=9.25
own_pred=own_pred=lr.predict([[9.25]])
print("The predicted score if a student studies for 9.25 hrs/day is",own_pred[0])


# In[27]:


hours=float(input())
own_pred=lr.predict([[hours]])
print("No of hours = {}". format(hours))
print("Predicted Score = {}". format(own_pred[0]))


# In[28]:


hours=float(input())
own_pred=lr.predict([[hours]])
print("No of hours = {}". format(hours))
print("Predicted Score = {}". format(own_pred[0]))

