import numpy as np
import pandas as pd


#Load the csv file as data frame.
df = pd.read_csv('C://Users//Marcell//Downloads//weather-dataset-rattle-package//weatherAUS.csv')
print("Size of weather data frame is:", df.shape)
print(df[0:5])

#Data preprocessing
#Checking for null values
print(df.count().sort_values())

df = df.drop(columns=['Sunshine','Evaporation','Cloud3pm','Cloud9am','Location','RISK_MM','Date'],axis=1)
print(df.shape)

df =df.dropna(how='any')
print(df.shape)

#Removes outliers/miscalculations in our data
from scipy import stats
z = np.abs(stats.zscore(df._get_numeric_data()))
print(z)
df = df[(z<3).all(axis=1)]
print(df.shape)

#replaces yes and no to zeros and ones
df['RainToday'].replace({'No':0,'Yes':1},inplace=True)
df['RainTomorrow'].replace({'No':0,'Yes':1},inplace=True)

#see unique values and covert to int
categorical_columns = ['WindGustDir','WindDir3pm','WindDir9am']
for col in categorical_columns:
    print(np.unique(df[col]))

#transform the categorical columns
df = pd.get_dummies(df,columns=categorical_columns)
print(df.iloc[4:9])

#standardize/normalize table data
from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler();
scaler.fit(df)
df = pd.DataFrame(scaler.transform(df),index=df.index,columns=df.columns)
print(df.iloc[4:10])

#Exploratory Data Analysis
from sklearn.feature_selection import SelectKBest,chi2
X = df.loc[:,df.columns!='RainTomorrow']
y = df[['RainTomorrow']]
selector = SelectKBest(chi2, k=3)
selector.fit(X,y)
X_new = selector.transform(X)
print(X.columns[selector.get_support(indices=True)])


#important features and assign them to variables
df = df[['Humidity3pm','Rainfall','RainTomorrow']]
X = df[['Humidity3pm']]
y = df[['RainTomorrow']]


#Data modelling
#Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time

#Calculating the accuracy and the time take by the classifiers
t0 = time.time()
#Data Splicing
#Logistic regression
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25)
clf_logreg = LogisticRegression(random_state=0)
#Building the model using the training data set
clf_logreg.fit(X_train,y_train.values.ravel())

#Evaluating the model using testing data set
y_pred = clf_logreg.predict(X_test)
score = accuracy_score(y_test,y_pred)

#Printing the accuracy and the time taken by the classifier
print('Accuracy using Logistic Regression:',score)
print('Time taken using Logistic Regression:', time.time()-t0)

#random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

#Calculating te accuracy and the time taken by the classifier
t0 = time.time()
#Random Forest Classifier
#Data Splicing
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25)
clf_rf = RandomForestClassifier(n_estimators=100,max_depth=4,random_state=0)
#Building the model using testing data set
clf_rf.fit(X_train,y_train.values.ravel())

#Evaluating the model using testing data set
y_pred = clf_rf.predict(X_test)
score = accuracy_score(y_test,y_pred)

#Printing the accuracy and the time taken by the classifier
print('Accuracy using Random Forest Classifier:',score)
print('Time taken using Random Forest Classifier:',time.time()-t0)

#Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

#Calculating the accuracy and the time take by the classifiers
t0= time.time()
#Data Splicing
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.25)
clf_dt = DecisionTreeClassifier(random_state=0)
#Building the model using the training data set
clf_dt.fit(X_train,y_train.values.ravel())

#Evaluating the model using testing data set
y_pred = clf_dt.predict(X_test)
score = accuracy_score(y_test,y_pred)

#Printing the accuracy and the time taken by the classifier
print('Accuracy using Decision Tree Classifier:', score)
print('Time taken using Decision Tree Classifier:',time.time()-t0)

#Support Vector Machine
from sklearn import svm
from sklearn.model_selection import train_test_split

#Calculating the accuracy and the time taken by the classifier
t0=time.time()
#Data Splicing
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25)
clf_svc = svm.SVC(kernel='linear')

#Building the model using the training data set
clf_svc.fit(X_train,y_train.values.ravel())

#Evaluating the model using testing data set
y_pred = clf_svc.predict(X_test)
score = accuracy_score(y_test,y_pred)

#Printing the accuracy and the time taken by the classifier
print('Accuracy using Support Vector Machine',score)
print('Time taken using Support Vector Machine',time.time()-t0)
