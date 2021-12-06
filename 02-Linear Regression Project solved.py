#!/usr/bin/env python
# coding: utf-8

# ___
# 
# <a href='http://www.pieriandata.com'> <img src='../Pierian_Data_Logo.png' /></a>
# ___
# # Linear Regression Project
# 
# Congratulations! You just got some contract work with an Ecommerce company based in New York City that sells clothing online but they also have in-store style and clothing advice sessions. Customers come in to the store, have sessions/meetings with a personal stylist, then they can go home and order either on a mobile app or website for the clothes they want.
# 
# The company is trying to decide whether to focus their efforts on their mobile app experience or their website. They've hired you on contract to help them figure it out! Let's get started!
# 
# Just follow the steps below to analyze the customer data (it's fake, don't worry I didn't give you real credit card numbers or emails).

# ## Imports
# ** Import pandas, numpy, matplotlib,and seaborn. Then set %matplotlib inline 
# (You'll import sklearn as you need it.)**

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Get the Data
# 
# We'll work with the Ecommerce Customers csv file from the company. It has Customer info, such as Email, Address, and their color Avatar. Then it also has numerical value columns:
# 
# * Avg. Session Length: Average session of in-store style advice sessions.
# * Time on App: Average time spent on App in minutes
# * Time on Website: Average time spent on Website in minutes
# * Length of Membership: How many years the customer has been a member. 
# 
# ** Read in the Ecommerce Customers csv file as a DataFrame called customers.**

# In[3]:


customers = pd.read_csv('Ecommerce Customers')


# **Check the head of customers, and check out its info() and describe() methods.**

# In[4]:


customers.head()


# In[5]:


customers.describe()


# In[6]:


customers.info()


# ## Exploratory Data Analysis
# 
# **Exploring the data!**
# 
# For the rest of the exercise we'll only be using the numerical data of the csv file.
# ___
# **Using seaborn to create a jointplot to compare the Time on Website and Yearly Amount Spent columns. Does the correlation make sense?**

# In[7]:


sns.jointplot(x='Time on Website',y='Yearly Amount Spent',data=customers)


# In[ ]:





# ** Doing the same but with the Time on App column instead. **

# In[8]:


sns.jointplot(x='Time on App',y='Yearly Amount Spent',data=customers)


# ** Using jointplot to create a 2D hex bin plot comparing Time on App and Length of Membership.**

# In[9]:


sns.jointplot(x='Time on App',y='Length of Membership',kind='hex', data=customers)


# **Exploring these types of relationships across the entire data set. Use [pairplot](https://stanford.edu/~mwaskom/software/seaborn/tutorial/axis_grids.html#plotting-pairwise-relationships-with-pairgrid-and-pairplot) to recreate the plot below.(Don't worry about the the colors)**

# In[10]:


sns.pairplot(customers)


# **Based off this plot what looks to be the most correlated feature with Yearly Amount Spent?**

# In[11]:


#Length of membership


# **Creating a linear model plot (using seaborn's lmplot) of  Yearly Amount Spent vs. Length of Membership. **

# In[12]:


sns.lmplot(x='Length of Membership',y='Yearly Amount Spent',data=customers)


# ## Training and Testing Data
# 
# Now that we've explored the data a bit, let's go ahead and split the data into training and testing sets.
# ** Set a variable X equal to the numerical features of the customers and a variable y equal to the "Yearly Amount Spent" column. **

# In[13]:


customers.columns


# In[14]:


X = customers[['Avg. Session Length', 'Time on App',
       'Time on Website', 'Length of Membership']]

y = customers['Yearly Amount Spent']


# ** Use model_selection.train_test_split from sklearn to split the data into training and testing sets. Set test_size=0.3 and random_state=101**

# In[15]:


from sklearn.model_selection import  train_test_split


# In[16]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# ## Training the Model
# 
# Training the model on our training data!
# 
# ** Import LinearRegression from sklearn.linear_model **

# In[28]:


from sklearn.linear_model import LinearRegression


# **Create an instance of a LinearRegression() model named lm.**

# In[29]:


lm = LinearRegression()


# ** Train/fit lm on the training data.**

# In[30]:


lm.fit(X_train,y_train)


# **Print out the coefficients of the model**

# In[27]:


print(lm.coef_)


# ## Predicting Test Data
# Fit the model, and evaluate its performance by predicting off the test values!
# 
# ** Use lm.predict() to predict off the X_test set of the data.**

# In[26]:


predictions = lm.predict(X_test)


# ** Create a scatterplot of the real test values versus the predicted values. **

# In[22]:


plt.scatter(y_test,predictions)
plt.xlabel('Y Test (True Values)')
plt.ylabel('Predicated Values')


# ## Evaluating the Model
# 
# Let's evaluate our model performance by calculating the residual sum of squares and the explained variance score (R^2).
# 
# ** Calculate the Mean Absolute Error, Mean Squared Error, and the Root Mean Squared Error. Refer to the lecture or to Wikipedia for the formulas**

# In[31]:


from sklearn import metrics
 


# In[33]:


print('MAE:', metrics.mean_absolute_error(y_test,predictions))
print('MSE:', metrics.mean_squared_error(y_test,predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test,predictions)))


# In[34]:


metrics.explained_variance_score(y_test,predictions)


# In[ ]:





# ## Residuals
 

# In[36]:


sns.distplot((y_test - predictions),bins=50)


# 

# ## Conclusion

# ** Recreate the dataframe below. **

# In[41]:


Outcome = pd.DataFrame(lm.coef_,X.columns,columns=['coeff'])


# In[42]:


Outcome





