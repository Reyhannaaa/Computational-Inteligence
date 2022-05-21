#!/usr/bin/env python
# coding: utf-8

# #### add important libraries and reading from dataset by pandas

# In[33]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[34]:


df = pd.read_csv("CRYPTOCURRENCY_COINDESK_BTCUSD_NEW.csv")


# #### using 'head' command to see data format

# In[36]:


df.tail(30)


# #### remove ',' char and convert it to float to work with them easily

# In[5]:


df['Close'] = df['Close'].str.replace(',','',regex=True).astype(int)
df


# # Direct solution

# #### add a column to our dataframe to observe the oscillation

# In[6]:


x = list(range(1,367))
x = np.array(x)
df['day'] = x
df.head()


# #### split our data to train and test(in this section we don't need valid set cuz it's linear)

# In[7]:


X = x


# In[8]:


y = np.array(df['Close'])


# In[9]:


def split(x, y , random_seed):
    x_train = list()
    x_test = list()
    y_train = list()
    y_test = list()
    for i in range(len(x)):
        if i % random_seed == 0:
            y_test.append(y[i])
            x_test.append(x[i])
        else:
            x_train.append(x[i])
            y_train.append(y[i])
    return x_train, y_train, x_test, y_test

X_train , y_train , X_test , y_test = split(X,y,5)

plt.scatter(X_train,y_train, color = 'blue')
plt.scatter(X_test,y_test , color = 'orange')


# In[10]:


X_train=np.array(X_train)
X_test=np.array(X_test)


# #### using w = ((X_train^T . X_train)^(-1)).(X_train^T).y_train to get the optimal weight

# In[11]:


X_train = X_train.reshape((len(X_train), 1))


# Optimal weight
w = np.linalg.inv(X_train.T.dot(X_train)).dot(X_train.T).dot(y_train)
w


# #### to find 'bias' i tried to compute the difference avg of y_train(that we had) and avg of ytrain(that is the predicted values by optimal w)

# In[12]:


ytrain = X_train.dot(w)
loss = np.mean(y_train) - np.mean(ytrain)
b = loss
print(b)


# #### predict ytest by optimal w and visualize it.

# In[13]:


X_test = X_test.reshape((len(X_test), 1))
ytest = X_test.dot(w)
 
diff = y_test - ytest
# y_pred = X.dot(w)
print(np.mean(diff))
plt.scatter(X, y , color='blue')
plt.plot(X_test, ytest + b , color='yellow')
plt.show()


# In[14]:


#### also check with sklearn model to see the difference


# In[15]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
X =  X.reshape((len(X), 1))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,
                                                    random_state=1)
model = LinearRegression()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)
plt.scatter(X, y,  color='blue')
plt.plot(X_test, y_pred, color='yellow')

print(model.coef_)
print(model.intercept_)
print(np.mean(y_test - y_pred))
# plt.scatter(X, y)
# plt.plot(X , model.coef_*X + model.intercept_)


# # Gradient descent

# #### split our data to train and test and visulize it.

# In[17]:


def split(x, y , random_seed):
    x_train = list()
    x_test = list()
    y_train = list()
    y_test = list()
    for i in range(len(x)):
        if i % random_seed == 0:
            y_test.append(y[i])
            x_test.append(x[i])
        else:
            x_train.append(x[i])
            y_train.append(y[i])
    return x_train, y_train, x_test, y_test

X_train , y_train , X_test , y_test = split(X,y,5)

plt.scatter(X_train,y_train, color = 'blue')
plt.scatter(X_test,y_test , color = 'orange')


# #### N is the number of iterations and at first we should consider value of w and bias zero and then update their values according to loss value(that used in computing dw and db) and then visualize it.
# #### i also print w , b and loss to see the changes.loss is going to be smaller..

# In[ ]:


w = 0
b = 0
N = 1000000
learning_rate = 0.0000001

X_train=np.array(X_train)
X_test=np.array(X_test)

for i in range(N):
    ypred = np.dot(w,X_train) + b
    loss = np.sum(((ypred - y_train)**2)/len(X_train))
    print(loss)
    dw = (-2/float(len(X_train)))*(np.sum(((y_train - ypred)*X_train)))
    db = (-2/float(len(X_train)))*(np.sum(((y_train - ypred))))
    
    w = w - learning_rate*dw
    b = b - learning_rate*db
    print(w,b)        
    
    

y_pred = np.dot(w,X_test) + b
plt.scatter(X, y) 
plt.plot(X_test,y_pred, color='red') 
plt.show()


# # Polynomial Regression

# #### read from csv file and split our data like before and this time we want valid data too.

# In[28]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("CRYPTOCURRENCY_COINDESK_BTCUSD_NEW.csv")

df['Close'] = df['Close'].str.replace(',','',regex=True).astype(int)
x = list(range(1,367))
x = np.array(x)
X = x 
y = np.array(df['Close'])
def split(x, y , random_seed):
    x_train = list()
    x_test = list()
    y_train = list()
    y_test = list()
    for i in range(len(x)):
        if i % random_seed == 0:
            y_test.append(y[i])
            x_test.append(x[i])
        else:
            x_train.append(x[i])
            y_train.append(y[i])
    return x_train, y_train, x_test, y_test

X_train , y_train , X_test , y_test = split(X,y,5)
X_train , y_train , X_valid , y_valid = split(X_train,y_train,5)

plt.scatter(X_train,y_train, color = 'blue')
plt.scatter(X_test,y_test , color = 'orange')
plt.scatter(X_valid,y_valid , color = 'yellow')


# ### polynomial implementation (Descriptions are given in the form of comments on the functions.)

# In[23]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



def split(x, y , random_seed):
    x_train = list()
    x_test = list()
    y_train = list()
    y_test = list()
    for i in range(len(x)):
        if i % random_seed == 0:
            y_test.append(y[i])
            x_test.append(x[i])
        else:
            x_train.append(x[i])
            y_train.append(y[i])
    return x_train, y_train, x_test, y_test



def fit(X,Y,degree) :
    
        # get dimension of X        
     
        m, n = X.shape
     
        # fist fill weight by zeros
     
        W = np.zeros( degree + 1 )
         
        # transform X for polynomial by dimension of X_train and the degree we wanted.
         
        X_transform = transform( X,m,degree )
         
        # normalize X_transform
         
        X_normalize = normalize( X_transform )
                 
        # try to find w according to loss that is difference of the predicted values and actual values for X_train.
     
        for i in range( N ) :
             
            h = predict( X , W , m , degree)
         
            loss = h - Y
             
            # update weights
         
            W = W - learning_rate * ( 1 / m ) * np.dot( X_normalize.T, loss )
      
        return W , m , degree
    
    
def transform( X,m,degree ) :
         
        # make an array by m dimension and fill it by 1
         
        X_transform = np.ones( ( m, 1 ) )
        # bring the function to a certain degree by empowering the data       
     
        for j in range(0 , degree + 1 ) :
             
            if j != 0 :
                 
                x_pow = np.power( X, j )
                 
                # append reshaped form of x_pow to X_transform
                 
                X_transform = np.append( X_transform, x_pow.reshape( -1, 1 ), axis = 1 )
 
        return X_transform  
        
    

def normalize( X ) :
    
        #  using stocastic normalize function to avoid gradient vanishing and exploding problems       
         
        X[:,1:] = ( X[:,1:] - np.mean( X[:,1:], axis = 0 ) ) / np.std( X[:,1:], axis = 0 )
         
        return X
    
def predict( X,W, m , degree ) :
      
        # transform X then normalize it to to multiply it by weight
         
        X_transform = transform( X,m,degree  )
         
        X_normalize = normalize( X_transform )
         
        return np.dot( X_transform, W )
    
 


# #### read again our data and split it first to train and test(20 percent of our data use for test) then split our train data to test and valid by the same amount.

# In[43]:


learning_rate = 0.01
N = 50000

df = pd.read_csv("CRYPTOCURRENCY_COINDESK_BTCUSD_NEW.csv")

df['Close'] = df['Close'].str.replace(',','',regex=True).astype(int)
x = list(range(1,367))
y = np.array(df['Close'])
x = np.array(x)
    
X_train , y_train , X_test , y_test = split(x,y,5)
X_test , y_test , X_valid , y_valid = split(X_test,y_test,2)


# use fit function to train our data by the degree we wanted and get the dimension of X_train and weight that is computed.
W , m , degree = fit( np.array(X_train).reshape(-1,1),y_train , 7 )


# want to predict y of valid data to see it's a good model or not.
# 37 is dimension of X_valid
yvalid = predict(np.array(X_valid) , W , 37 , degree)

# if amount was small our model is good enough but if it was big we may encounter overfitting or underfitting.
print('loss for valid data: ')
print(np.mean(yvalid - y_valid))

# 37 is dimension of X_test
# predict y for X_test
y_pred = predict( np.array(X_test) , W , 37 , degree)

X_10_next_days = list(range(367,377))
y_10_next_days = predict(np.array(X_10_next_days) , W , 10 , degree)
y_10_next_days = y_10_next_days.astype(int)
print('predicted values for next 10 days: ')
print(y_10_next_days)
     
plt.scatter( x, y, color = 'blue' )
     
plt.plot( np.array(X_test), y_pred , color = 'red' )
     
plt.show()


# ### 23 is enough small value for loss of valid data.
# ### the predicted values computed and printed.

# In[ ]:




