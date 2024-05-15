#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv('../DATA/cancer_classification.csv')


# In[3]:


df.info()


# In[4]:


df.describe()


# In[5]:


df.describe().transpose()


# In[6]:


sns.countplot(x='benign_0__mal_1',data=df)


# In[7]:


df.corr()


# In[8]:


df.corr()['benign_0__mal_1'].sort_values()


# In[9]:


df.corr()['benign_0__mal_1'].sort_values().plot(kind='bar')


# In[10]:


df.corr()['benign_0__mal_1'][:-1].sort_values().plot(kind='bar')


# In[11]:


plt.figure(figsize=(12,12))
sns.heatmap(df.corr())


# In[12]:


X = df.drop('benign_0__mal_1',axis=1).values


# In[13]:


y = df['benign_0__mal_1'].values


# In[14]:


from sklearn.model_selection import train_test_split


# In[15]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=101)


# In[16]:


from sklearn.preprocessing import MinMaxScaler


# In[17]:


scaler = MinMaxScaler()


# In[18]:


X_train = scaler.fit_transform(X_train)


# In[19]:


X_test = scaler.transform(X_test)


# In[21]:


from tensorflow.keras.models import Sequential


# In[23]:


from tensorflow.keras.layers import Dense,Dropout


# In[25]:


X_train.shape


# In[28]:


model = Sequential()

model.add(Dense(30,activation='relu'))
model.add(Dense(15,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam')


# In[29]:


model.fit(x=X_train,y=y_train,epochs=600,validation_data=(X_test,y_test))


# In[30]:


losses = pd.DataFrame(model.history.history)


# In[31]:


losses.plot()


# In[32]:


model = Sequential()

model.add(Dense(30,activation='relu'))
model.add(Dense(15,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam')


# In[33]:


from tensorflow.keras.callbacks import EarlyStopping


# In[34]:


early_stop = EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=25)


# In[35]:


model.fit(x=X_train,y=y_train,epochs=600,validation_data=(X_test,y_test),
         callbacks=[early_stop])


# In[36]:


model_loss = pd.DataFrame(model.history.history)
model_loss.plot()


# In[42]:


model = Sequential()

model.add(Dense(30,activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(15,activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam')


# In[43]:


model.fit(x=X_train,y=y_train,epochs=600,validation_data=(X_test,y_test),
         callbacks=[early_stop])


# In[46]:


model.loss = pd.DataFrame(model.history.history)
model_loss.plot()


# In[48]:


predictions = (model.predict(X_test) > 0.5)*1 


# In[49]:


from sklearn.metrics import classification_report,confusion_matrix


# In[50]:


print(classification_report(y_test,predictions))


# In[51]:


print(confusion_matrix(y_test,predictions))


# In[ ]:




