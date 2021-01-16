# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 09:12:44 2020

@author: Anup0
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score,classification_report,confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

path="D:/Nikhil Analytics/Python/Project/New project/"
Ccard_data=pd.read_csv(path + "creditcard.csv",header=0)
Ccard_data.head()
Ccard_data.shape
Ccard_data.describe()
Ccard_data.isnull().sum()


sns.countplot(Ccard_data.Class,label="class",color="red")

# Target variable has value 0 very large compared to value 1.

target0=Ccard_data[Ccard_data.Class==0]
print(len(target0))
target1=Ccard_data[Ccard_data.Class==1]
print(len(target1))

balanced_data=pd.concat([target1,target0.sample(n=len(target1),random_state=10)])
#Doubt1


sns.countplot(balanced_data.Class,label="class",color="green")

balanced_data.describe()
#Doubt2

# Now target variable is balanced

X=balanced_data.iloc[:,:-1]
Y=balanced_data.iloc[:,-1]

train_x,test_x,train_y,test_y=train_test_split(X,Y,test_size=0.3,random_state=10)


#USING RANDOM FOREST:-

random_model=RandomForestClassifier(n_estimators=20,criterion="entropy")
random_model.fit(train_x,train_y)
pred_y=random_model.predict(test_x)

print(roc_auc_score(test_y, pred_y)) #0.9388668218965438
print(accuracy_score(test_y,pred_y))  #0.9391891891891891
print(confusion_matrix(test_y,pred_y))
#[[147   2]
#[ 16 131]]
print(classification_report(test_y,pred_y))

#           0       0.90      0.99      0.94       149
#           1       0.98      0.89      0.94       147

# Using logistic regression:-

logistic=LogisticRegression(random_state=10).fit(train_x,train_y)
pred_y=logistic.predict(test_x)

print(accuracy_score(test_y,pred_y)) #0.902027027027027
print(confusion_matrix(test_y,pred_y))
# [[138  11]
# [ 18 129]]
print(classification_report(test_y,pred_y))
#           0       0.88      0.93      0.90       149
#           1       0.92      0.88      0.90       147


# Using Gradient Boosting clasifier:- 

gbm_model=GradientBoostingClassifier(random_state=10).fit(train_x,train_y)
pred_y=gbm_model.predict(test_x)  

print(accuracy_score(test_y,pred_y)) #0.9256756756756757
print(confusion_matrix(test_y,pred_y))
#[[144   5]
 #[ 17 130]]
print(classification_report(test_y,pred_y))
#           0       0.89      0.97      0.93       149
#           1       0.96      0.88      0.92       147

"""
Project is about CREDIT CARD fraud transactions which I did using python, 
packages used are pandas,seaborn and sklearn.
Started by extracting data in pandas dataframe and checking for null values, which were absent.
Checked for unbalanced data by plotting count graph of target variable.
As the data was unbalanced, balanced the data by concatinating.
Proceeded and declared feature and target variables, using new balanced dataset. (Didn't drop any columns') 
I put 'class' as Y, then splitted the data set into train and test with test size 30%
Since, it is an unbalanced dataset, I used random forest  first to create model.
I used classifier instead of regressor because data is descrete.
Criterion as entropy instead of gini for more accuracy.
Finally checked the accuracy of model by comparing test sample of y with predicted y using the model.
Since the data is unbalanced, used roc_auc_score to measure the accuracy of the model. 
Using the above parameters , I got an acuuracy of - 0.9388668218965438 (best fit)
I also created model using gradient boosting and logistic regression.
The best accuracy was achieved by random forest.
"""
 
 








