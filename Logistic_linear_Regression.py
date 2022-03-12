#!/usr/bin/env python
# coding: utf-8

# In[161]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge,Lasso,RidgeCV,LassoCV,ElasticNet,LogisticRegression
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import accuracy_score,confusion_matrix,roc_auc_score,roc_curve
import matplotlib.pyplot as plt
from pandas_profiling import ProfileReport
import seaborn as sns
import pickle


# In[162]:


df=pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv")
#Loaded the dataSet


# In[163]:


ProfileReport(df)#Analyising the Dataset


# In[164]:


df['BMI']=df['BMI'].replace(0,df['BMI'].mean())#Handling the zero values present in the column


# In[165]:


df['Glucose']=df['Glucose'].replace(0,df['Glucose'].mean())#Handling the zero values present in the column


# In[166]:


df['BloodPressure']=df['BloodPressure'].replace(0,df['BloodPressure'].mean())#Handling the zero values present in the column


# In[167]:


df['SkinThickness']=df['SkinThickness'].replace(0,df['SkinThickness'].mean())#Handling the zero values present in the column


# In[168]:


df['Insulin']=df['Insulin'].replace(0,df['Insulin'].mean())#Handling the zero values present in the column


# In[169]:


fig,ax=plt.subplots(figsize=(20,20))#As we have scew data in our dataset so we are ploting the graph
sns.boxplot(data=df,ax=ax)#to showcase the outliers


# In[170]:


q=df['Insulin'].quantile(.95)
df_new=df[df['Insulin']<q]#Treatments for outliers


# In[171]:


fig,ax=plt.subplots(figsize=(20,20))#seeing the graph again
sns.boxplot(data=df_new,ax=ax)


# In[172]:


#keep on checking with various quantile untill you get min outlier or zero outlier


# In[173]:


q=df['Pregnancies'].quantile(.98)
df_new=df[df['Pregnancies']<q]#Treatments for outliers


# In[174]:


q=df['BloodPressure'].quantile(.99)
df_new=df[df['BloodPressure']<q]#Treatments for outliers


# In[175]:


q=df['SkinThickness'].quantile(.99)
df_new=df[df['SkinThickness']<q]#Treatments for outliers


# In[176]:


q=df['Insulin'].quantile(.95)
df_new=df[df['Insulin']<q]#Treatments for outliers


# In[177]:


q=df['DiabetesPedigreeFunction'].quantile(.95)
df_new=df[df['DiabetesPedigreeFunction']<q]#Treatments for outliers


# In[178]:


q=df['Age'].quantile(.99)
df_new=df[df['Age']<q]#Treatments for outliers


# In[179]:


ProfileReport(df_new)#for analysing the dataset again ,still we got an outlier but its better than previous one
#One very important question may occurs from here is that how you are dealing with outlier so the answer will be by 
#changing quantile range


# In[180]:


y=df_new['Outcome']#putting the label or outcome in y


# In[181]:


x=df_new.drop(columns=['Outcome'])#putting the feature in x


# In[182]:


scaler=StandardScaler()
x_Scaled=scalar.fit_transform(x)


# In[183]:


def vif_score(x):
    scaler = StandardScaler()
    arr = scaler.fit_transform(x)#Normalizing the data
    return pd.DataFrame([[x.columns[i], variance_inflation_factor(arr,i)] for i in range(arr.shape[1])], columns=["FEATURE", "VIF_SCORE"])#with same time finding correlation


# In[184]:


vif_score(x)#No Value is greater then 10 so no correlation found if we found any so we will try to remove the column


# In[185]:


x_train,x_test,y_train,y_test=train_test_split(x_Scaled,y,test_size=.20,random_state=144)#Spliting the data


# In[224]:


y_train


# In[187]:


logr=LogisticRegression()#making the object of class


# In[188]:


logr.fit(x_train,y_train)#training the dataset


# In[189]:


logr.score(x_train,y_train)#giving the accuracy


# In[190]:


logr.predict([x_test[0]])#predict the values


# In[191]:


logr.predict_proba([x_test[0]])#predict the probability for 1 class to 2 class which aprox 94% for one class and for 5% for second class


# In[192]:


y.iloc[0]#here we are getting the y as 1 but we are getting the predict values as 0


# In[193]:


#now lets predict with some other solver


# In[194]:


logr_lib=LogisticRegression(solver="liblinear")#creating the new logr_lib object


# In[195]:


logr=LogisticRegression()#creating the new logr class


# In[196]:


logr_lib.fit(x_train,y_train)#fiting the dataset for logr_lib


# In[197]:


logr.fit(x_train,y_train)#fiting the dataset min logr


# In[301]:


logr_lib.predict([x_test[0]])#predicting the values logr for one values


# In[302]:


logr.predict([x_test[0]])#predicitng the values with logr for one values


# In[303]:


logr.score(x_test,y_test)


# In[304]:


logr_lib.score(x_test,y_test)


# In[202]:


#Now for the whole dataset predict the values of dataset and store in any var 


# In[305]:


y_predict_logr_lib=logr_lib.predict(x_test)


# In[306]:


y_predict_logr=logr.predict(x_test)


# In[307]:


confusion_matrix(y_test,y_predict_logr_lib)#creating the confusion matrix


# In[308]:


confusion_matrix(y_test,y_predict_logr)


# In[309]:


tn,fp,fn,tp=confusion_matrix(y_test,y_predict_logr_lib).ravel()
accuracy=(tp+tn)/(tp+tn+fp+fn)
precision=tp/(tp+fp)
recall=tp/(tp+fn)
specificity=tn/(fp+tn)
F1_Score = 2*(recall * precision) / (recall + precision)
result={"Accuracy":accuracy,"Precision":precision,"Recall":recall,'Specficity':specificity,'F1':F1_Score}
result


# In[310]:


tn,fp,fn,tp=confusion_matrix(y_test,y_predict_logr).ravel()
accuracy=(tp+tn)/(tp+tn+fp+fn)
precision=tp/(tp+fp)
recall=tp/(tp+fn)
specificity=tn/(fp+tn)
F1_Score = 2*(recall * precision) / (recall + precision)
result={"Accuracy":accuracy,"Precision":precision,"Recall":recall,'Specficity':specificity,'F1':F1_Score}
result


# In[311]:


roc_auc_score(y_test,y_predict_logr)


# In[312]:


roc_auc_score(y_test,y_predict_logr_lib)


# In[316]:


auc=fpr, tpr, thresholds  = roc_curve(y_test,y_predict_logr_lib)#from here we got fpr,tpr,thresholds


# In[314]:


fpr, tpr, thresholds  = roc_curve(y_test,y_predict_logr)


# In[317]:


plt.plot(fpr, tpr, color='orange', label='ROC')
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--',label='ROC curve (area = %0.2f)' % auc)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




