import matplotlib.pyplot as plt
import pandas as pd
import pyodbc
# Environment settings: 
pd.set_option('display.max_column', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_seq_items', None)
pd.set_option('display.max_colwidth', 500)
pd.set_option('expand_frame_repr', True)
defaults.to_csv('base_table_' + name_string + '.csv', index=False)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.metrics import f1_score
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
# mlp for multi-label classification
from numpy import mean
from numpy import std
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import RepeatedKFold
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import accuracy_score
# get the dataset
def get_dataset():
	X = defaults.drop(columns = ['vie', 'du', 'tr', 'ket', 'pen', 'ses', 'sep', 'ast'])
	X.shape
	y = defaults.drop(columns = ['Navigation_deep','Navigation_skip_overview','Forum_visit','Forum_post','Video_pictures' ,'Content_text_stay'
	,'Feedback_no','NO_connections_links','Quiz_revisions','Ques_detail','Ques_facts','Ques_concepts'])
	y.shape
	return X, y
# load dataset
X, y = get_dataset()
X.describe()
y.describe()
# create dataframe from file
dataframe = X
# use corr() method on dataframe to
# make correlation matrix
matrix = dataframe.corr()
# print correlation matrix
print("Correlation Matrix is : ")
print(matrix)
corr = dataframe.corr()
corr.style.background_gradient(cmap='coolwarm')
from keras.models import Sequential
from keras.layers import Dense, Activation
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import label_binarize
n_inputs, n_outputs = X.shape[1], y.shape[1]
print(n_inputs, n_outputs)
# Binarize the output
y = label_binarize(y, classes=['vie', 'du', 'tr', 'ket', 'pen', 'ses', 'sep', 'ast'])
n_classes = y.shape[1]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from lightgbm import LGBMClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
import matplotlib.pylab as pl
from sklearn.metrics import roc_curve, auc
# Learn to predict each class against the other
classif = OneVsRestClassifier(SVC(kernel='linear'))
classif =classif.fit(X_train, y_train)
yhat = classif.predict(X_test)
print(yhat)
y_score = classif.decision_function(X_test)
print(y_score)
# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
n_classes = y.shape[1]
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
# Plot of a ROC curve for a specific class
#Receiver Operating Characteristic (ROC) metric to evaluate classifier output quality.
plt.figure()
plt.plot(fpr[4], tpr[4], label='ROC curve (area = %0.2f)' % roc_auc[4])
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
# Plot of a ROC curve for a specific class
#Receiver Operating Characteristic (ROC) metric to evaluate classifier output quality.
plt.figure()
plt.plot(fpr[5], tpr[5], label='ROC curve (area = %0.2f)' % roc_auc[5])
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
# Plot ROC curve
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]))
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'
                                   ''.format(i, roc_auc[i]))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()
from sklearn.metrics import hamming_loss, accuracy_score 
y_true = y_test
print (y_true)
y_pred = yhat
print (y_pred)
print("accuracy_score:", accuracy_score(y_true, y_pred))
print("Hamming_loss:", hamming_loss(y_true, y_pred))
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score

m = MultiLabelBinarizer().fit(y_true)

f1_score(m.transform(y_true),
         m.transform(y_pred),
         average='macro')
from sklearn.metrics import precision_recall_fscore_support
precision_recall_fscore_support(y_true, y_pred, average='macro')
precision_recall_fscore_support(y_true, y_pred, average='micro')
precision_recall_fscore_support(y_true, y_pred, average='weighted')
precision_recall_fscore_support(y_true, y_pred)
from sklearn.metrics import classification_report
print(classification_report(y_true, y_pred, labels=[0, 1, 2, 3, 4, 5, 6, 7 ]))
from sklearn.linear_model import Perceptron
clf= OneVsRestClassifier(Perceptron(tol=1e-3, random_state=0))
clf=clf.fit(X_train, y_train)
yhatt = clf.predict(X_test)
print(yhatt)
#Predict margin (libsvm name for this is predict_values)
y_scoree = clf.decision_function(X_test)
# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
n_classes = y.shape[1]
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_scoree[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_scoree.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
# Plot of a ROC curve for a specific class
#Receiver Operating Characteristic (ROC) metric to evaluate classifier output quality.
plt.figure()
plt.plot(fpr[4], tpr[4], label='ROC curve (area = %0.2f)' % roc_auc[4])
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
# Plot of a ROC curve for a specific class
#Receiver Operating Characteristic (ROC) metric to evaluate classifier output quality.
plt.figure()
plt.plot(fpr[5], tpr[5], label='ROC curve (area = %0.2f)' % roc_auc[5])
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
# Plot ROC curve
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]))
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'
                                   ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()
from sklearn.metrics import hamming_loss, accuracy_score 
y_true = y_test
print (y_true)
y_pred = yhatt
print (y_pred)
print("accuracy_score:", accuracy_score(y_true, y_pred))
print("Hamming_loss:", hamming_loss(y_true, y_pred))
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score
m = MultiLabelBinarizer().fit(y_true)
f1_score(m.transform(y_true),
         m.transform(y_pred),
         average='macro')
from sklearn.metrics import precision_recall_fscore_support
precision_recall_fscore_support(y_true, y_pred, average='macro')
precision_recall_fscore_support(y_true, y_pred, average='micro')
precision_recall_fscore_support(y_true, y_pred, average='weighted')
precision_recall_fscore_support(y_true, y_pred)
from sklearn.metrics import classification_report
print(classification_report(y_true, y_pred, labels=[0, 1, 2, 3, 4, 5, 6, 7 ]))
from sklearn.linear_model import LogisticRegression
clf= OneVsRestClassifier(LogisticRegression(random_state=0, max_iter=600))
clf=clf.fit(X_train, y_train)
yhattt = clf.predict(X_test)
print(yhatt)
#Predict margin (libsvm name for this is predict_values)
y_scoreee = clf.decision_function(X_test)
# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
n_classes = y.shape[1]
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_scoreee[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_scoreee.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
# Plot of a ROC curve for a specific class
#Receiver Operating Characteristic (ROC) metric to evaluate classifier output quality.
plt.figure()
plt.plot(fpr[4], tpr[4], label='ROC curve (area = %0.2f)' % roc_auc[4])
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
# Plot of a ROC curve for a specific class
#Receiver Operating Characteristic (ROC) metric to evaluate classifier output quality.
plt.figure()
plt.plot(fpr[5], tpr[5], label='ROC curve (area = %0.2f)' % roc_auc[5])
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
# Plot ROC curve
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]))
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'
                                   ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()
from sklearn.metrics import hamming_loss, accuracy_score 
y_true = y_test
print (y_true)
y_pred = yhattt
print (y_pred)
print("accuracy_score:", accuracy_score(y_true, y_pred))
print("Hamming_loss:", hamming_loss(y_true, y_pred))
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score
m = MultiLabelBinarizer().fit(y_true)
f1_score(m.transform(y_true),
         m.transform(y_pred),
         average='macro')
from sklearn.metrics import precision_recall_fscore_support
precision_recall_fscore_support(y_true, y_pred, average='macro')
from sklearn.metrics import precision_recall_fscore_support
precision_recall_fscore_support(y_true, y_pred, average='micro')
from sklearn.metrics import precision_recall_fscore_support
precision_recall_fscore_support(y_true, y_pred, average='weighted')
precision_recall_fscore_support(y_true, y_pred)
from sklearn.metrics import classification_report
print(classification_report(y_true, y_pred, labels=[0, 1, 2, 3, 4, 5, 6, 7 ]))
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import hamming_loss
from sklearn.linear_model import Perceptron
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
def hamming_score(y_true, y_pred, normalize=True, sample_weight=None):
    acc_list = []
    for i in range(y_true.shape[0]):
        set_true = set( np.where(y_true[i])[0] )
        set_pred = set( np.where(y_pred[i])[0] )
        tmp_a = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            tmp_a = len(set_true.intersection(set_pred))/float(len(set_true.union(set_pred)))
        acc_list.append(tmp_a)
    return np.mean(acc_list)
def print_score(y_pred, clf):
    print("Clf: ", clf.__class__.__name__)
    print("Hamming loss: {}".format(hamming_loss(y_pred, y_test)))
    print("Hamming score: {}".format(hamming_score(y_pred, y_test)))
    print("---")    
nb_clf = MultinomialNB()
sgd = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=6, tol=None)
lr = LogisticRegression(random_state=0, max_iter=900)
mn = LinearSVC(random_state=0,max_iter=130000, tol=1e-5)
prc = Perceptron(tol=1e-3, random_state=0)
bst = GradientBoostingClassifier(n_estimators=200, learning_rate=1.0, max_depth=1, random_state=0)
pag =PassiveAggressiveClassifier(max_iter=1000, random_state=0,tol=1e-5)

for classifier in [nb_clf, sgd, lr, mn, prc,bst,pag]:
    clf = OneVsRestClassifier(classifier)
    clf.fit(X_train, y_train)
    y_predd = clf.predict(X_test)
    print_score(y_predd, classifier)
