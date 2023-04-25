#!/usr/bin/env python
# coding: utf-8

# # Machine Learning (ML) Project By :
# 
# Name :Shweta Ganesh Dhotre
# 
# Email id :shwetapawar1811@gmail.com ,Contact no:7775992772
# 
# Linkedin:www.linkedin.com/in/shweta-dhotre-407282255

# # Predict Bankrupt Country using Machine Learning:

# In[1]:


#import all required libraries

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier,GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report, accuracy_score


# # EDA AND VISUALIZATION

# In[2]:


##Read path and converted into csv:

df=pd.read_csv(r"C:\Users\Saksh\Qualitative_Bankruptcy.data.txt",header=None)
df.head()


# In[3]:


#Naming a attrtibutes:
columns=["Industrial Risk","Management Risk","Financial Flexibility","Credibility","Competitiveness","Operating Risk","Class"]
df.columns=columns


# In[4]:


df.isna().sum()


# In[5]:


df.info()


# In[7]:


#Read the csv format data in Jupyter GUI:
df


# In[8]:


df.info()


# In[9]:


df.head()


# In[34]:


df.tail()


# In[35]:


df.describe()


# In[6]:


from sklearn import preprocessing 
lb=preprocessing.LabelEncoder()

df["Industrial Risk"]=lb.fit_transform(df["Industrial Risk"])
df["Management Risk"]=lb.fit_transform(df["Management Risk"])
df["Financial Flexibility"]=lb.fit_transform(df["Financial Flexibility"])
df["Credibility"]=lb.fit_transform(df["Credibility"])
df["Competitiveness"]=lb.fit_transform(df["Competitiveness"])
df["Operating Risk"]=lb.fit_transform(df["Operating Risk"])
df["Class"]=lb.fit_transform(df["Class"])


# # VISUALIZATION

# In[36]:


#visualization

plt.figure(figsize=(9,7))
sns.heatmap(df.corr(), annot=True)
plt.show()


# In[38]:


#Histogram of Age Distribution:
plt.hist(df['Industrial Risk'], bins=10, edgecolor='black')
plt.xlabel('Industrial Risk')
plt.ylabel('Frequency')
plt.title('Distribution of Industrial Risk')
plt.show()


# In[39]:


#Correlation Heatmap:
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Heatmap')
plt.show()


# In[40]:


#Barplot of Target Variable:
 
plt.figure(figsize=(6,4))
sns.countplot(x='Class', data=df)
plt.xlabel('Class')
plt.ylabel('Count')
plt.title('Distribution of Class')
plt.show()


# In[43]:


#Boxplot of Management Risk by Class Variable:
plt.figure(figsize=(3,2))
sns.boxplot(x='Class', y='Industrial Risk', data=df)
plt.xlabel('Class')
plt.ylabel('Indusrial Risk')
plt.title('Industrial Risk by Class')
plt.show()


# In[45]:



#Pairplot of Selected Features:
sns.pairplot(data=df[['Industrial Risk', 'Management Risk', 'Financial Flexibility', 'Operating Risk', 'Competitiveness',  'Class']], hue='Class')
plt.show()


# In[49]:


# Visualize the data with boxplots:
cols = ['Class', 'Operating Risk', 'Competitiveness', 'Credibility', 'Financial Flexibility', 'Industrial Risk']
rows = ['Managment Risk', 'Financial Flexibility', 'Credibility', 'Competitiveness', 'Operating Risk', 'Class']
fig, axs = plt.subplots(nrows=len(rows), ncols=len(cols), figsize=(22, 22))
for row in range(len(rows)):
    for col in range(len(cols)):
        sns.boxplot(x='Class', y=cols[col], data=df, ax=axs[row, col])
        axs[row, col].set_title(f"{rows[row].title()} vs {cols[col].title()}")
plt.tight_layout()
plt.show()


# In[50]:


#We will see the skewness of all the numerical columns in our dataset by a distplot:
plt.figure(figsize=(8,8))
plt.suptitle("Visualizing the distribution of skewness in Class",fontsize=10)
plotnumber=1
for column in df:
    if plotnumber<=13:
        ax=plt.subplot(4,4,plotnumber)
        sns.distplot(df[column],color="blue")
        plt.xlabel(column,fontsize=14)
    plotnumber+=1
plt.tight_layout()


# In[51]:



#We will now see the Outliers present in our Numerical columns through Boxplot:
plt.figure(figsize=(8,8))
plt.suptitle("Visualizing the outliers through boxplot",fontsize=10)
plotnumber=1
for column in df:
    if plotnumber<=12:
        ax=plt.subplot(4,3,plotnumber)
        sns.boxplot(df[column],color="red")
        plt.xlabel(column,fontsize=14)
    plotnumber+=1
plt.tight_layout()


# In[10]:


x = df.iloc[:,:-1]
x


# In[11]:


y=df.iloc[:,-1]
y


# # CREATING A ML MODEL

# In[12]:


#Splitting the dataset into training and testing data:
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3,random_state=3)


# # DEVELOPING MODEL

# In[13]:


def mymodel(model):
 model.fit(xtrain,ytrain)
 ypred=model.predict(xtest)
 
 train=model.score(xtrain,ytrain)
 test=model.score(xtest,ytest)
 
 print(f"Training Accuracy: {train}\nTesting Accuracy: {test}\n\n")
 print(classification_report(ytest,ypred))
 
 return model


# In[ ]:


xtrain


# # Gaussian Naive Bayes classifier:

# In[14]:


nb = GaussianNB()
print("\nGaussian Naive Bayes:")
nb = mymodel(GaussianNB())


# # Decision Tree classifier:
# 

# In[15]:


dt = DecisionTreeClassifier()
print("\nDecision Tree:")
dt = mymodel( DecisionTreeClassifier())


# # Random Forest classifier:

# In[16]:


rf = RandomForestClassifier()
print("\nRandom Forest:")
rf = mymodel(RandomForestClassifier())


# # Support Vector Machine Classifier:

# In[17]:


svm = SVC()
print("\nSupport Vector Machine:")
svm = mymodel(SVC())


# # LOGISTIC REGRESSION

# In[18]:


logreg=LogisticRegression()
print("\nLogistic Regression:")
logreg = mymodel(LogisticRegression())


# # KNeighbors Classifier:
# 

# In[19]:


knn= KNeighborsClassifier()
print("\nKNeighbors Classifier:")
knn = mymodel(KNeighborsClassifier())


# # XGB CLASSIFIER

# In[20]:


get_ipython().system('pip install xgboost')
from xgboost import XGBClassifier
xgb =XGBClassifier()
print("\nXGB Classifier:")
xgb = mymodel(XGBClassifier())


# In[21]:


xgb = mymodel(XGBClassifier(max_depth=2))


# # CROSS VALIDATION

# In[22]:


from sklearn.model_selection import cross_val_score


# In[23]:


# Define the models:
models = [('Log.Regression',LogisticRegression()),
 ('G.NaiveBayes', GaussianNB()),
 ('Decision Tree', DecisionTreeClassifier()),
 ('Random Forest', RandomForestClassifier()),
 ('XGBClassifier1', XGBClassifier()),
 ('XGBClassifier2', XGBClassifier(max_depth=2)),
 ('SVM ', SVC()),
 ('K-NNClassifier',KNeighborsClassifier()),
 ]


# In[24]:


results = []
for name, model in models:
 scores = cross_val_score(model, x, y, cv=5)
 results.append((name, scores.mean(), scores.std()))


# In[25]:


print("Model\t\t\tAccuracy\tStandard deviation")
print("--------------------------------------------------------")
for name, mean, std in results:
 print(f"{name}\t\t{mean:.4f}\t\t{std:.4f}")


# In[26]:


from sklearn.model_selection import KFold, cross_val_score
clf = RandomForestClassifier(random_state=1)
k_folds = KFold(n_splits = 5)
scores = cross_val_score(clf, x, y, cv = k_folds)
print("Cross Validation Scores: ", scores)
print("Average CV Score: ", scores.mean())
print("Number of CV Scores used in Average: ", len(scores))


# In[27]:


model_1 = RandomForestClassifier(random_state=1)
model_1.fit(xtrain, ytrain)
ypred = model_1.predict(xtest)
train = model_1.score(xtrain, ytrain)
test = model_1.score(xtest, ytest)
print(f"Training Accuracy : {train}\nTesting Accuracy : {test}\n\n")
print(classification_report(ytest, ypred))


# In[28]:


from sklearn.metrics import roc_auc_score
from sklearn.metrics import RocCurveDisplay
RocCurveDisplay.from_predictions(ytest, ypred)
print(f'model AUC score: {roc_auc_score(ytest, ypred)}')


# In[29]:


from sklearn.model_selection import KFold, cross_val_score
clf = XGBClassifier(random_state=1)
k_folds = KFold(n_splits = 5)
scores = cross_val_score(clf, x, y, cv = k_folds)
print("Cross Validation Scores: ", scores)
print("Average CV Score: ", scores.mean())
print("Number of CV Scores used in Average: ", len(scores))


# In[30]:


model_2 = XGBClassifier(random_state=1)
model_2.fit(xtrain, ytrain)
ypred = model_2.predict(xtest)
train = model_2.score(xtrain, ytrain)
test = model_2.score(xtest, ytest)
print(f"Training Accuracy : {train}\nTesting Accuracy : {test}\n\n")
print(classification_report(ytest, ypred))


# In[31]:


from sklearn.metrics import roc_auc_score
from sklearn.metrics import RocCurveDisplay
RocCurveDisplay.from_predictions(ytest, ypred)
print(f'model 1 AUC score: {roc_auc_score(ytest, ypred)}')


# In[32]:


ypred_RF = model_2.predict(xtest)
y_score_RF = model_2.predict_proba(xtest)[:,1]


# In[33]:


from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
model_2_roc_auc = roc_auc_score(ytest, model_2.predict(xtest))
fpr, tpr, thresholds = roc_curve(ytest, model_2.predict_proba(xtest)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='LDA (area = %0.2f)' % model_2_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([-0.01, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




