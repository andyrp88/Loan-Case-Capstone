#!/usr/bin/env python
# coding: utf-8

# <a href="https://www.bigdatauniversity.com"><img src="https://ibm.box.com/shared/static/cw2c7r3o20w9zn8gkecaeyjhgw3xdgbj.png" width="400" align="center"></a>
# 
# <h1 align="center"><font size="5">Classification with Python</font></h1>

# In this notebook we try to practice all the classification algorithms that we learned in this course.
# 
# We load a dataset using Pandas library, and apply the following algorithms, and find the best one for this specific dataset by accuracy evaluation methods.
# 
# Lets first load required libraries:

# In[1]:


import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing
get_ipython().run_line_magic('matplotlib', 'inline')


# ### About dataset

# This dataset is about past loans. The __Loan_train.csv__ data set includes details of 346 customers whose loan are already paid off or defaulted. It includes following fields:
# 
# | Field          | Description                                                                           |
# |----------------|---------------------------------------------------------------------------------------|
# | Loan_status    | Whether a loan is paid off on in collection                                           |
# | Principal      | Basic principal loan amount at the                                                    |
# | Terms          | Origination terms which can be weekly (7 days), biweekly, and monthly payoff schedule |
# | Effective_date | When the loan got originated and took effects                                         |
# | Due_date       | Since it’s one-time payoff schedule, each loan has one single due date                |
# | Age            | Age of applicant                                                                      |
# | Education      | Education of applicant                                                                |
# | Gender         | The gender of applicant                                                               |

# Lets download the dataset

# In[2]:


get_ipython().system('wget -O loan_train.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/loan_train.csv')


# ### Load Data From CSV File  

# In[3]:


df = pd.read_csv('loan_train.csv')
df.head()


# In[4]:


df.shape


# ### Convert to date time object 

# In[4]:


df['due_date'] = pd.to_datetime(df['due_date'])
df['effective_date'] = pd.to_datetime(df['effective_date'])
df.head()


# # Data visualization and pre-processing
# 
# 

# Let’s see how many of each class is in our data set 

# In[5]:


df['loan_status'].value_counts()


# 260 people have paid off the loan on time while 86 have gone into collection 
# 

# Lets plot some columns to underestand data better:

# In[6]:


# notice: installing seaborn might takes a few minutes
get_ipython().system('conda install -c anaconda seaborn -y')


# In[7]:


import seaborn as sns

bins = np.linspace(df.Principal.min(), df.Principal.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'Principal', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()


# In[8]:


bins = np.linspace(df.age.min(), df.age.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'age', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()


# # Pre-processing:  Feature selection/extraction

# ### Lets look at the day of the week people get the loan 

# In[9]:


df['dayofweek'] = df['effective_date'].dt.dayofweek
bins = np.linspace(df.dayofweek.min(), df.dayofweek.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'dayofweek', bins=bins, ec="k")
g.axes[-1].legend()
plt.show()


# We see that people who get the loan at the end of the week dont pay it off, so lets use Feature binarization to set a threshold values less then day 4 

# In[10]:


df['weekend'] = df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)
df.head()


# ## Convert Categorical features to numerical values

# Lets look at gender:

# In[11]:


df.groupby(['Gender'])['loan_status'].value_counts(normalize=True)


# 86 % of female pay there loans while only 73 % of males pay there loan
# 

# Lets convert male to 0 and female to 1:
# 

# In[12]:


df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)
df.head()


# ## One Hot Encoding  
# #### How about education?

# In[13]:


df.groupby(['education'])['loan_status'].value_counts(normalize=True)


# #### Feature befor One Hot Encoding

# In[14]:


df[['Principal','terms','age','Gender','education']].head()


# #### Use one hot encoding technique to conver categorical varables to binary variables and append them to the feature Data Frame 

# In[15]:


Feature = df[['Principal','terms','age','Gender','weekend']]
Feature = pd.concat([Feature,pd.get_dummies(df['education'])], axis=1)
Feature.drop(['Master or Above'], axis = 1,inplace=True)
Feature.head()


# ### Feature selection

# Lets defind feature sets, X:

# In[16]:


X = Feature
X[0:5]


# What are our lables?

# In[17]:


y = df['loan_status'].values
y[0:5]


# ## Normalize Data 

# Data Standardization give data zero mean and unit variance (technically should be done after train test split )

# In[18]:


X= preprocessing.StandardScaler().fit(X).transform(X)
X[0:5]


# # Classification 

# Now, it is your turn, use the training set to build an accurate model. Then use the test set to report the accuracy of the model
# You should use the following algorithm:
# - K Nearest Neighbor(KNN)
# - Decision Tree
# - Support Vector Machine
# - Logistic Regression
# 
# 
# 
# __ Notice:__ 
# - You can go above and change the pre-processing, feature selection, feature-extraction, and so on, to make a better model.
# - You should use either scikit-learn, Scipy or Numpy libraries for developing the classification algorithms.
# - You should include the code of the algorithm in the following cells.

# # K Nearest Neighbor(KNN)
# Notice: You should find the best k to build the model with the best accuracy.  
# **warning:** You should not use the __loan_test.csv__ for finding the best k, however, you can split your train_loan.csv into train and test to find the best __k__.

# In[57]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)


# In[58]:


Ks = 10
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))
ConfustionMx = [];
for n in range(1,Ks):
    
    #Train Model and Predict  
    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
    yhat=neigh.predict(X_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)
    std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])
mean_acc


# In[59]:


plt.plot(range(1,Ks),mean_acc,'g')
plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
plt.legend(('Accuracy ', '+/­ 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Nabors (K)')
plt.tight_layout()
plt.show()
print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1)


# In[60]:


k = 7
#Train Model and Predict  
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
yhat = neigh.predict(X_test)
yhat[0:5]


# # Decision Tree

# In[41]:


from sklearn.tree import DecisionTreeClassifier
loanTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
loanTree


# In[42]:


X_train03, X_test03, y_train03, y_test03 = train_test_split( X, y, test_size=0.3, random_state=3)
print ('Train set:', X_train03.shape,  y_train03.shape)
print ('Test set:', X_test03.shape,  y_test03.shape)

loanTree.fit(X_train03,y_train03)
predTree = loanTree.predict(X_test03)

print (predTree [0:5])
print (y_test [0:5])


# In[43]:


from sklearn import metrics
import matplotlib.pyplot as plt
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_test03, predTree))


# # Support Vector Machine

# In[44]:


from sklearn import svm
from sklearn.metrics import f1_score
from sklearn.metrics import jaccard_similarity_score


# In[45]:


clf = svm.SVC(kernel='rbf')
clf.fit(X_train, y_train)
yhat = clf.predict(X_test)
print (yhat [0:5])
print (y_test [0:5])


# In[46]:


print("Avg F1-score: %.4f" % f1_score(y_test, yhat, average='weighted'))
print("Jaccard score: %.4f" % jaccard_similarity_score(y_test, yhat))


# # Logistic Regression

# In[47]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
LogR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,y_train)
LogR


# In[48]:


yhat = LogR.predict(X_test)
print (yhat [0:10])
print (y_test [0:10])


# In[49]:


from sklearn.metrics import jaccard_similarity_score
print ("Jaccard similarity score: : %.4f" % jaccard_similarity_score(y_test, yhat))

from sklearn.metrics import log_loss
yhat_prob = LogR.predict_proba(X_test)
print ("LogLoss: : %.4f" % log_loss(y_test, yhat_prob))


# # Model Evaluation using Test set

# In[50]:


from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss


# First, download and load the test set:

# In[51]:


get_ipython().system('wget -O loan_test.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/loan_test.csv')


# ### Load Test set for evaluation 

# In[52]:


test_df = pd.read_csv('loan_test.csv')
test_df.head()


# In[53]:


test_df['effective_date'] = pd.to_datetime(test_df['effective_date'])
test_df['dayofweek'] = test_df['effective_date'].dt.dayofweek
test_df['weekend'] = test_df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)
test_df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)
Feature2 = test_df[['Principal','terms','age','Gender','weekend']]
Feature2 = pd.concat([Feature2,pd.get_dummies(test_df['education'])], axis=1)
Feature2.drop(['Master or Above'], axis = 1,inplace=True)
X2 = Feature2
X2 = preprocessing.StandardScaler().fit(X2).transform(X2)
X2[0:5]
X2.shape


# In[54]:


y2 = test_df['loan_status'].values
y2[0:5]
y2.shape


# In[61]:


# k nearest neighbours
yhat_knn = neigh.predict(X2)
jacscore_knn = jaccard_similarity_score(y2, yhat_knn)
f1score_knn=f1_score(y2, yhat_knn, average='weighted')
print("Jaccard score: %.4f" % jacscore_knn)
print("Avg F1-score: %.4f" % f1score_knn)


# In[62]:


# decision tree
yhat_tree = loanTree.predict(X2)
jacscore_tree = jaccard_similarity_score(y2, yhat_tree)
f1score_tree=f1_score(y2, yhat_tree, average='weighted')
print("Jaccard score: %.4f" % jacscore_tree)
print("Avg F1-score: %.4f" % f1score_tree)


# In[63]:


#svm
yhat_svm = clf.predict(X2)
jacscore_svm = jaccard_similarity_score(y2, yhat_svm)
f1score_svm=f1_score(y2, yhat_svm, average='weighted')
print("Jaccard score: %.4f" % jacscore_svm)
print("Avg F1-score: %.4f" % f1score_svm)


# In[64]:


#Logistic Rregression
yhat_lr = LogR.predict(X2)
jacscore_lr = jaccard_similarity_score(y2, yhat_lr)
f1score_lr=f1_score(y2, yhat_lr, average='weighted')
print("Jaccard score: %.4f" % jacscore_lr)
print("Avg F1-score: %.4f" % f1score_lr)
yhat_prob_lr = LogR.predict_proba(X2)
logloss_lr = log_loss(y2, yhat_prob_lr)
print ("LogLoss: %.4f" % logloss_lr)


# In[65]:


from IPython.display import Markdown as md
md("| Algorithm | Jaccard  | F1-score  | LogLoss |\n| ------ | ------- | -------- | ------- |\n| KNN    | %.4f    |  %.4f   |    NA    |\n| Decision Tree | %.4f | %.4f | NA |\n| SVM | %.4f | %.4f | NA |\n| LogisticRegression | %.4f | %.4f | %.4f |"
   % (jacscore_knn, f1score_knn, jacscore_tree, f1score_tree, jacscore_svm, f1score_svm, jacscore_lr, f1score_lr, logloss_lr))


# # Report
# You should be able to report the accuracy of the built model using different evaluation metrics:

# | Algorithm          | Jaccard | F1-score | LogLoss |
# |--------------------|---------|----------|---------|
# | KNN                | ?       | ?        | NA      |
# | Decision Tree      | ?       | ?        | NA      |
# | SVM                | ?       | ?        | NA      |
# | LogisticRegression | ?       | ?        | ?       |

# <h2>Want to learn more?</h2>
# 
# IBM SPSS Modeler is a comprehensive analytics platform that has many machine learning algorithms. It has been designed to bring predictive intelligence to decisions made by individuals, by groups, by systems – by your enterprise as a whole. A free trial is available through this course, available here: <a href="http://cocl.us/ML0101EN-SPSSModeler">SPSS Modeler</a>
# 
# Also, you can use Watson Studio to run these notebooks faster with bigger datasets. Watson Studio is IBM's leading cloud solution for data scientists, built by data scientists. With Jupyter notebooks, RStudio, Apache Spark and popular libraries pre-packaged in the cloud, Watson Studio enables data scientists to collaborate on their projects without having to install anything. Join the fast-growing community of Watson Studio users today with a free account at <a href="https://cocl.us/ML0101EN_DSX">Watson Studio</a>
# 
# <h3>Thanks for completing this lesson!</h3>
# 
# <h4>Author:  <a href="https://ca.linkedin.com/in/saeedaghabozorgi">Saeed Aghabozorgi</a></h4>
# <p><a href="https://ca.linkedin.com/in/saeedaghabozorgi">Saeed Aghabozorgi</a>, PhD is a Data Scientist in IBM with a track record of developing enterprise level applications that substantially increases clients’ ability to turn data into actionable knowledge. He is a researcher in data mining field and expert in developing advanced analytic methods like machine learning and statistical modelling on large datasets.</p>
# 
# <hr>
# 
# <p>Copyright &copy; 2018 <a href="https://cocl.us/DX0108EN_CC">Cognitive Class</a>. This notebook and its source code are released under the terms of the <a href="https://bigdatauniversity.com/mit-license/">MIT License</a>.</p>
