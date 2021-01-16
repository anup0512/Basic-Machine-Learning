# Basic-Machine-Learning-Projects:
1. Credit Card Fraud Detection:
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
