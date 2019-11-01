### Code by Adonis Silva Flores
### Special thanks to the developer of pandas making big data processing simpler, scipy making memory used efficient and scikit-learn making machine learning easy!!!

# importing the required libraries/modules
from tkinter import *
from tkinter import filedialog
import os, sys
import numpy as np
import pandas as pd
from IPython.display import display
import scipy
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.naive_bayes import ComplementNB
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier)
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_curve, roc_auc_score
from matplotlib import pyplot as plt
from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer 

Tk().withdraw()
# Open file to transform (csv files)
file_path = filedialog.askopenfilename(initialdir="C:/Users/Eigenaar/Desktop/python_prog",
                                       filetypes=[('.csvfiles', '.csv')],
                                       title='Select csv file')

filename=os.path.split(file_path)[1]
print ("\nData loaded: {}".format(filename))

input_data = file_path

# comma delimited is the default
df = pd.read_csv(input_data, header=0, low_memory=False)
                
# for space delimited use:
## df = pd.read_csv(input_file, header = 0, delimiter = " ")

# for tab delimited use:
## df = pd.read_csv(input_file, header = 0, delimiter = "\t")

columns = df.columns.tolist()
dfin = pd.read_csv(input_data, usecols = columns[:len(columns)-1], low_memory=False)
dftar = pd.read_csv(input_data, usecols = columns[len(columns)-1:len(columns)], low_memory=False)

################################################################################################
# join text data ...
sys.__stdout__ = sys.stdout   # just to fix some bug ....
dftext = (dfin['project_title'].map(str)+' '+df['project_essay_1'].map(str)+' '+df['project_essay_2'].map(str)+' '
          +df['project_essay_3'].map(str)+' '+df['project_essay_4'].map(str)+' '+df['project_resource_summary'].map(str))

# remove the text data from the orig file ...
dfin = dfin.drop(['project_title','project_essay_1', 'project_essay_2', 'project_essay_3', 'project_essay_4', 'project_resource_summary'],axis=1)
# transform categorical features ... one-hot encoding
dfin=pd.get_dummies(dfin)
# impute values on missing entries (mean)...
dfin=dfin.fillna(value=dfin.mean())

print('categorical variables transformed ...')

# put the original column names in a python list
original_headers = list(df.columns.values)
# put the numeric column names in a python list
numeric_headers = list(df.columns.values)

# create a numpy array for input into scikit-learn
idata = dfin.as_matrix()
itext = dftext.as_matrix()
itarget = dftar.as_matrix()

X_data=idata
X_text=itext
y_target=itarget[:182080]

# transform features containing text data ... bag of words (result is scipy sparse matrix)
class LemmaTokenizer(object):
    print('applying lemmatization ...')
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]
vec = TfidfVectorizer(ngram_range=(1, 1), tokenizer=LemmaTokenizer())
X_text = vec.fit_transform(X_text)
print('text data transformed ...')
# X_data converted to scipy sparse matrix as well ...
X_data=scipy.sparse.csc_matrix(X_data)
# rejoin transformed orig and text data ...
X_join=hstack([X_data,X_text]).tocsr()

# split training data from new data to be classified ...
X_datatrain=X_join[:182080,:]
X_class=X_join[182080:,:]

# split X_datatrain into training (and validation) and test sets, default 75% training and 25% test set
X_train, X_test, y_train, y_test = train_test_split(X_datatrain, y_target, test_size=0.25, random_state=42)

# training and testing model ...
clf = GradientBoostingClassifier(n_estimators=500, learning_rate=0.2, verbose=2)
#clf = RandomForestClassifier(n_estimators=100, verbose=2)
# fit the model to training data
print('training in progress ...')
clf.fit(X_train, y_train.ravel())
print('training finished ...')
# evaluate the model using the test data
y_pred = clf.predict_proba(X_test)[:, 1]

# Plot the PR curve ...
precision_clf, recall_clf, thresholds_ = precision_recall_curve(y_test, y_pred)
AvePre = average_precision_score(y_test, y_pred)
fig1=plt.figure(figsize=(6, 5))
plt.plot(recall_clf, precision_clf, 'C2', label='AvePrecision={0:0.2f}'.format(AvePre))
plt.step(recall_clf, precision_clf, color='g', alpha=0.2, where='mid')
plt.fill_between(recall_clf, precision_clf, interpolate=True, step='mid', color='g', alpha=0.2)

plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.xscale('linear')
plt.title('Precision-Recall Curve, GradientBoosting')
plt.legend()
fig1.show()

# Plot ROC curve ...
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
AUC = roc_auc_score(y_test, y_pred)
fig2=plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, 'C2', label='AUC={0:0.2f}'.format(AUC))
plt.step(fpr, tpr, color='g', alpha=0.2, where='mid')
plt.fill_between(fpr, tpr, interpolate=True, step='mid', color='g', alpha=0.2)

plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.xscale('linear')
plt.title('ROC Curve, GradientBoosting')
plt.legend()
fig2.show()

###########################################################################################################
# Predict class probability of new data ...

# produce predictions ...
out=np.column_stack((clf.predict_proba(X_class),clf.predict(X_class)))
np.set_printoptions(precision=3, suppress=True)

hdr=[]
for i in range(1, len(out[0])):
    hd="Pclass{}".format(i-1)
    hdr.append(hd)

pc="Predicted class"
hdr.append(pc)
hdrs=", ".join(map(str, hdr))

print("Class Probabilities and Predictions:")
datas=pd.DataFrame(out)
datas.columns=[hdr]
pd.options.display.float_format = '{:10,.2f}'.format
display(datas)

# save predictions ...
np.savetxt('C:/Users/Eigenaar/Desktop/python_prog/donorpred.csv', out, fmt='%0.2f', comments='',
            header=hdrs, delimiter=",")

print("Class prediction also saved!!!")





