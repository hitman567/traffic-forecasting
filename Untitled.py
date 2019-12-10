
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


data = pd.read_csv("traffic_data.csv")


# In[3]:


data.head()


# In[4]:


data1 = data.iloc[:,1:4]
data2 = data.iloc[:,4:7]
data3 = data.iloc[:,7:10]
data4 = data.iloc[:,10:13]
data5 = data.iloc[:,13:16]
data6 = data.iloc[:,16:19]


# In[5]:


data6.head()


# In[6]:


data1['date'] = pd.date_range('11/01/1995','31/12/1997')
data2['date'] = pd.date_range('11/01/1995','31/12/1997')
data3['date'] = pd.date_range('11/01/1995','31/12/1997')
data4['date'] = pd.date_range('11/01/1995','31/12/1997')
data5['date'] = pd.date_range('11/01/1995','31/12/1997')
data6['date'] = pd.date_range('11/01/1995','31/12/1997')


# In[7]:


data1.head()


# In[282]:


plt.figure(figsize=(10,6))
# plt.scatter(pd.date_range('11/01/1995','31/12/1997'),data4.iloc[:,0],marker='X')
plt.scatter(pd.date_range('11/01/1995','31/12/1997'),data6.iloc[:,0],marker='x')
# plt.scatter(pd.date_range('11/01/1995','31/12/1997'),data5.iloc[:,0],marker='+')
plt.xlabel('Date')
plt.ylabel('Total traffic')


# In[9]:


data1['day'] = data1['date'].dt.day
data1['month'] = data1['date'].dt.month
data1['year'] = data1['date'].dt.year


# In[10]:


data2['day'] = data1['date'].dt.day
data2['month'] = data1['date'].dt.month
data2['year'] = data1['date'].dt.year


# In[11]:


data3['day'] = data1['date'].dt.day
data3['month'] = data1['date'].dt.month
data3['year'] = data1['date'].dt.year


# In[12]:


data4['day'] = data1['date'].dt.day
data4['month'] = data1['date'].dt.month
data4['year'] = data1['date'].dt.year


# In[13]:


data5['day'] = data1['date'].dt.day
data5['month'] = data1['date'].dt.month
data5['year'] = data1['date'].dt.year


# In[14]:


data6['day'] = data1['date'].dt.day
data6['month'] = data1['date'].dt.month
data6['year'] = data1['date'].dt.year


# In[15]:


data1 = data1.drop('date',axis=1)


# In[16]:


data2 = data2.drop('date',axis=1)


# In[17]:


data3 = data3.drop('date',axis=1)


# In[18]:


data4 = data4.drop('date',axis=1)


# In[19]:


data5 = data5.drop('date',axis=1)


# In[20]:


data6 = data6.drop('date',axis=1)


# In[283]:


data1.head()


# In[21]:


# groupbyDay1 = data1.groupby('day')
# groupbyMonth1 = data1.groupby('month') 
# groupbyYear1 = data1.groupby('year')


# In[22]:


# groupbyDay2 = data2.groupby('day')
# groupbyMonth2 = data2.groupby('month') 
# groupbyYear2 = data2.groupby('year')


# In[23]:


# groupbyDay3 = data3.groupby('day')
# groupbyMonth3 = data3.groupby('month') 
# groupbyYear3 = data3.groupby('year')


# In[24]:


# groupbyDay4 = data4.groupby('day')
# groupbyMonth4 = data4.groupby('month') 
# groupbyYear4 = data4.groupby('year')


# In[25]:


# groupbyDay5 = data5.groupby('day')
# groupbyMonth5 = data5.groupby('month') 
# groupbyYear5 = data5.groupby('year')


# In[26]:


# groupbyDay6 = data6.groupby('day')
# groupbyMonth6 = data6.groupby('month') 
# groupbyYear6 = data6.groupby('year')


# In[27]:


# groupbyDay1.first()
# groupbyMonth1.first()
# groupbyYear1.first()


# In[28]:


# groupbyDay2.first()
# groupbyMonth2.first()
# groupbyYear2.first()


# In[29]:


# groupbyDay3.first()
# groupbyMonth3.first()
# groupbyYear3.first()


# In[30]:


# groupbyDay4.first()
# groupbyMonth4.first()
# groupbyYear4.first()


# In[31]:


# groupbyDay5.first()
# groupbyMonth5.first()
# groupbyYear5.first()


# In[32]:


# groupbyDay6.first()
# groupbyMonth6.first()
# groupbyYear6.first()


# In[126]:


X = data1.iloc[:,1:].values
y = data1.iloc[:,0].values


# In[34]:


# plt.figure(figsize=(10,6))
# plt.scatter(data1['day'],y,color='red')
sns.jointplot(x=data1['month'],y=y,data=data1,kind='scatter')


# In[35]:


sns.jointplot(x=data1['day'],y=y,data=data1,kind='scatter')


# In[127]:


X


# In[128]:


y


# In[129]:


y = y.reshape(-1,1)


# In[130]:


from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler


# In[131]:


scaler = MinMaxScaler(feature_range=(0,1))


# In[132]:


X = scaler.fit_transform(X)
y = scaler.fit_transform(y)


# In[133]:


X


# In[134]:


y


# In[135]:


seed = 42
X,y = shuffle(X,y, random_state=seed)


# In[136]:


para = {'C':[0.1,1,100,1000],
       'epsilon':[0.0001,0.0005,0.001,0.005,0.01,0.05,0.1,0.5,1,5,10],
       'gamma':[0.0001,0.001,0.005,0.1,1,3,5]}


# In[137]:


gsc = GridSearchCV(estimator=SVR(kernel='rbf'),param_grid=para,cv=5, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)


# In[138]:


from sklearn.model_selection import train_test_split


# In[139]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)


# In[140]:


# grid_result = gsc.fit(X_train,y_train)
grid_result = gsc.fit(X,y)


# In[249]:


best_params = grid_result.best_params_
best_predict = grid_result.predict(X_test)
# print(best_predict)
print(best_params)


# In[250]:


best_svr = SVR(kernel='rbf',C=best_params['C'],epsilon=best_params['epsilon'],
              gamma=best_params['gamma'],coef0=0.1)


# In[251]:


scoring = {'abs_eror':'neg_mean_absolute_error',
          'squared_error':'neg_mean_squared_error'}


# In[252]:


scores = cross_validate(best_svr,X,y,cv=10,scoring=scoring,return_train_score=True)


# In[253]:


scores


# In[254]:


abs(scores['test_abs_eror'].mean())


# In[255]:


math.sqrt(abs(scores['test_squared_error'].mean()))


# In[256]:


best_svr.fit(X_train,y_train)


# In[257]:


pred = best_svr.predict(X_test)
# print(pred)


# In[150]:


best_svr.score(X_test,y_test)


# In[151]:


from sklearn.metrics import r2_score


# In[152]:


r2_score(y_test,best_predict)


# In[62]:


best_predict.reshape(-1,1)


# In[284]:


bus=1750
multi=1000
day=2
month=11
year=1995


# In[285]:


ans= best_svr.predict(scaler.transform(np.array([[bus,multi,day,month,year]])))


# In[286]:


ans


# In[287]:


y_pred = scaler.inverse_transform([ans])


# In[288]:


y_pred


# In[276]:


# plt.scatter(pd.date_range('11/01/1995','31/12/1997'), data1.iloc[:,0], color='darkorange', label='data')
# plt.hold('on')
# plt.plot(pd.date_range('11/01/1995','31/12/1997',best_predict, color='navy', lw=lw, label='RBF model')
# plt.xlabel('data')
# plt.ylabel('target')
# plt.title('Support Vector Regression')
# plt.legend()
# plt.show()

