# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 10:18:50 2021
@author: marcocappai
"""
#Data Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from matplotlib import rc
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, confusion_matrix, plot_confusion_matrix
from sklearn import metrics
from sklearn.pipeline import Pipeline

plt.style.use('seaborn')
rc('text', usetex=True)
warnings.filterwarnings("ignore")
#%%
#Importing the data-set and dropping nan values
#df = pd.read_csv('dataset.csv').dropna()
df = pd.read_excel('dataCOMP0050Coursework1.xlsx').dropna()
df = df.loc[:, df.columns != 'issue_d']
#Plotting a bar plot of the binary outcome
default_0 = len(df[df['charged_off'] == 0])
default_1 = len(df[df['charged_off'] == 1])
sns.barplot(x=['0', '1'], y=[default_0, default_1],
            alpha=0.7, edgecolor='k')
plt.ylabel('frequency', fontsize=16)
plt.xlabel('charged off', fontsize=16)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.title('Unbalanced dependent variable',
          fontsize=25)
#%% Resampling dataset
count_class_0, count_class_1 = df['charged_off'].value_counts()
df_class_0 = df[df['charged_off'] == 0]
df_class_1 = df[df['charged_off'] == 1]

df_class_0_under = df_class_0.sample(count_class_1)
df = pd.concat([df_class_0_under, df_class_1], axis=0)

palette = ['red', 'blue']
sns.barplot(x=['0', '1'], y=[len(df_class_0_under), len(df_class_1)],
            alpha=0.7, edgecolor='k', palette=palette)
plt.ylabel('frequency', fontsize=16)
plt.xlabel('charged off', fontsize=16)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.title('Rebalanced dependent variable',
          fontsize=25)
#-----------------------------------------------------------------------------------------
#%% Two graphs merged as subplots
palette = ['red', 'blue']
plt.rc('xtick',labelsize=14)
plt.rc('ytick',labelsize=14)
fig, axs = plt.subplots(1,2, figsize=(14,5))
sns.barplot(x=['0', '1'], y=[default_0, default_1],
            alpha=0.7, edgecolor='k', 
            ax=axs[0], palette=palette)
axs[0].set_ylabel('frequency', fontsize=22)
axs[0].set_xlabel('charged off', fontsize=22)
axs[0].set_title('Unbalanced dependent variable',
          fontsize=22)
sns.barplot(x=['0', '1'], y=[len(df_class_0_under), len(df_class_1)],
            alpha=0.7, edgecolor='k', palette=palette,
            ax=axs[1])
axs[1].set_ylabel('frequency', fontsize=22)
axs[1].set_xlabel('charged off', fontsize=22)
axs[1].set_title('Rebalanced dependent variable',
          fontsize=22)

#%%
fig, axs = plt.subplots(2,2, figsize=(14,12))
#-----------------------------------------------------------------------------------------
#[0]
sns.distplot(df['charged_off'], kde=False,
            bins=2,
            hist_kws=dict(edgecolor="k", linewidth=1),
            ax=axs[0,0], color='lightgrey')
axs[0,0].grid(linewidth=0.5)
axs[0,0].set_xlabel('charged off')
axs[0,0].set_ylabel('frequency')
axs[0,0].set_xlim([0,1.01])
axs[0,0].set_xticks(axs[0,0].get_xticks()[::5])
#-----------------------------------------------------------------------------------------
#[1]
sns.distplot(df['loan_amnt'], kde=False,
            hist_kws=dict(edgecolor="k", linewidth=1),
            bins=15, ax=axs[0,1],
            color='pink', label='loan amount')
axs[0,1].set_xlabel('loan amount')
axs[0,1].set_ylabel('frequency')
axs[0,1].set_xticks(axs[0,1].get_xticks()[::2])
axs[0,1].grid(linewidth=0.5)
axs[0,1].set_xlim([0,np.max(df['loan_amnt'])])
axs[0,1].legend()
#-----------------------------------------------------------------------------------------
#[2]
sns.distplot(df['log_annual_inc'], kde=False,
            hist_kws=dict(edgecolor="k", linewidth=1),
            bins=15, ax=axs[1,0],
            color='lightgreen', label='annual income (log)')
axs[1,0].set_xlabel('log income')
axs[1,0].set_ylabel('frequency')
axs[1,0].grid(linewidth=0.5)
axs[1,0].set_xlim(np.min(df['log_annual_inc']), 
                         np.max(df['log_annual_inc']))
axs[1,0].legend()
#-----------------------------------------------------------------------------------------
#4
sns.distplot(df['fico_score'], kde=False,
            hist_kws=dict(edgecolor="k", linewidth=1),
            bins=15, ax=axs[1,1],
            color='lightblue', label='fico score')
axs[1,1].set_xlabel('fico score')
axs[1,1].set_ylabel('frequency')
axs[1,1].grid(linewidth=0.5)
axs[1,1].legend()
#%% Dummy Variables
df = pd.get_dummies(df, columns=['home_ownership', 
                                 'verification_status',
                                'application_type',
                                'purpose'], drop_first=True)
df = df.loc[:, df.columns != 'earliest_cr_line']
#%% Splitting data into training and cross val
X = df.loc[:, df.columns != 'charged_off'] #matrix of features
y = df['charged_off'] #vectors of outcomes
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=0, 
                                                    stratify=y)
#%% HELPER FUNCTIONS
def learning_curves(model, X, y):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
    train_errors = np.array([])
    test_errors = np.array([])
    for i in range(10, 200):
        model.fit(x_train[:i], y_train[:i])
        y_train_predict = model.predict(x_train[:i])
        y_test_predict = model.predict(x_test)
        train_errors = np.append(train_errors, 
                                roc_auc_score(y_train[:i],
                                             y_train_predict))
        test_errors = np.append(test_errors,
                               roc_auc_score(y_test,
                                            y_test_predict))
    plt.plot(train_errors, 'r-', linewidth=1.5, label='training',
            marker='*')
    plt.plot(test_errors, 'b-', linewidth=1.5, label='test')
    plt.xlabel('size of training set')
    plt.ylabel('AUC score')
    plt.grid(linewidth=0.5)
    plt.legend()
#----------------------------------------------------------------------------------------
def compute_score(model, X, y,
                 cv_n=5, scoring='roc_auc'):
    k = KFold(n_splits=cv_n, shuffle=True, random_state=0)
    return cross_val_score(model, X, y,
                          scoring=scoring, cv=k).mean()
def compute_score_f1(model, X, y,
                     cv_n=5, scoring='f1'):
    k = KFold(n_splits=cv_n, shuffle=True, random_state=0)
    return cross_val_score(model, X, y,
                           scoring=scoring, cv=k).mean()
def compute_score_accuracy(model, X, y,
                           cv_n=5, scoring='accuracy'):
    k = KFold(n_splits=cv_n, shuffle=True, random_state=0)
    return cross_val_score(model, X, y,
                           scoring=scoring, cv=k).mean()
#----------------------------------------------------------------------------------------
def plot_roc_graph(y_test, preds, color='blue', name=''):
    fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
    roc_auc = metrics.auc(fpr, tpr)
    plt.figure(figsize=(6,6))
    plt.title('ROC curve '+name,
              fontsize=22)
    plt.plot(fpr, tpr, label='AUC {} = {}'.format(name ,round(roc_auc, 3)),
            color=color, linewidth=1)
    plt.plot([0,1], [0,1], 'r-', linewidth=1,
            label='benchmark')
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.ylabel('True positive rate',
               fontsize=20)
    plt.xlabel('False positive rate',
               fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(linewidth=0.5)
    plt.legend(fontsize=14)
    plt.show()
#----------------------------------------------------------------------------------------    
def compute_testscores(y_test, y_pred, y_prob):
    auc_score = roc_auc_score(y_test, y_prob)
    acc_score = accuracy_score(y_test, y_pred)
    score_f1 = f1_score(y_test, y_pred)
    return [auc_score, acc_score, score_f1]
#----------------------------------------------------------------------------------------
#%% Scaling data through standard scaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)

#%% FEATURE SELECTION
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE

k_features = np.arange(1,X.shape[1]+1,1)
string_array = [str(i) for i in k_features] #turning list of integers into strings

def select_useful_features(X_scaled, y, k_features=k_features):
    auc_scores = np.array([])
    accuracy_scores_RFE = []
    f1_scores_RFE = []
    for i in k_features:
        model = LogisticRegression(random_state=0)
        selector = RFE(model, n_features_to_select=i, 
                         step=True, verbose=True)
        selector.fit(X_scaled, y)
        X_selected = selector.transform(X)
        auc_score = compute_score(model, X_selected, y)
        auc_scores = np.append(auc_scores, auc_score)
        accuracy_score = compute_score_accuracy(model, X_selected, y)
        accuracy_scores_RFE.append(accuracy_score)
        f1_score = compute_score_f1(model, X_selected, y)
        f1_scores_RFE.append(f1_score)
    scores_df = pd.DataFrame(auc_scores, index=k_features,
                             columns=['scores'])
    max_value = scores_df[max(scores_df.values)==scores_df.values].index.values[0]
    plt.plot(string_array, auc_scores, linewidth=1, color='black',
             marker='o', markersize=7, label='AUC')   
    plt.plot(string_array, accuracy_scores_RFE, linewidth=1,
             marker='o', markersize=7, label='accuracy',
             color='red')
    plt.plot(string_array, f1_scores_RFE, linewidth=1,
             marker='o', markersize=7, label='f1')
    plt.axvline(x=10, color='red', linewidth=1,
                linestyle='--', label='chosen with {} features'.format(11))
    plt.ylabel('scoring', fontsize=20)
    plt.xlabel('number of features selected', fontsize=20)
    plt.title('Recursive feature elimination', fontsize=30)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(linewidth=0.5)
    plt.legend(fontsize=14)
    
select_useful_features(X_scaled, y) # we now know that 15 is the optimal features, so
#we run and make a separate dataset with just those features

model = LogisticRegression()
selector = RFE(model, n_features_to_select=11,
               step=True, verbose=True)
selector.fit(X_scaled, y)
column_names = X.columns
params = selector.get_support()
selected_columns = column_names[params]
print('The selected columns are {}'.format(selected_columns))
X_selected = selector.transform(X)

#Re-splitting the dataset
X_selected_train, X_selected_test, y_train, y_test = train_test_split(X_selected, y, 
                                                    test_size=0.2, 
                                                    random_state=0, 
                                                    stratify=y)
#%% LOGISTIC REGRESSION
logit_model = Pipeline([
    ('scaler', StandardScaler()),
    ('logit_clf', LogisticRegression())
])
accuracy_score_logit = compute_score_accuracy(logit_model, X_selected_train, 
                                              y_train)
auc_score_logit = compute_score(logit_model, X_selected_train, 
                                y_train)
f1_score_logit = compute_score_f1(logit_model, X_selected_train, 
                                  y_train)

#Testing the model on the test data
logit_model.fit(X_selected_train, y_train)
y_pred_logit = logit_model.predict_proba(X_selected_test)[:,1]
plot_roc_graph(y_test, y_pred_logit, name='Logistic Regression',
               color='black')

y_predictions = model.predict(X_selected_test)
cm_logit = confusion_matrix(y_test, y_predictions)

scores_logit = compute_testscores(y_test, y_predictions, y_pred_logit)

#%% K-NEAREST NEIGHBORS
from sklearn.neighbors import KNeighborsClassifier
k = [5, 10, 20, 50, 100, 200, 300, 
     400, 500, 600, 700,1000]
string_array_k = [str(i) for i in k] 

scores_KNN_auc = []
scores_KNN_accuracy = []
scores_KNN_f1 = []
for i in k:
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('knn_clf', KNeighborsClassifier(n_neighbors=i,
                                metric='minkowski',
                                p=2))
    ])
    score = compute_score(model, X_train, y_train)
    scores_KNN_auc.append(score)
    score_accuracy = compute_score_accuracy(model, X_train, y_train)
    scores_KNN_accuracy.append(score_accuracy)
    score_f1 = compute_score_f1(model, X_train, y_train)
    scores_KNN_f1.append(score_f1)
    
plt.figure(figsize=(7,6))
plt.plot(string_array_k, 
         scores_KNN_auc, marker='o', linewidth=1, 
         markersize=5, label='AUC')
plt.plot(string_array_k, 
         scores_KNN_accuracy, marker='o', linewidth=1, 
         markersize=5, label='accuracy')
plt.plot(string_array_k, 
         scores_KNN_f1, marker='o', 
         linewidth=1, markersize=5, label='f1')
plt.xlabel('number of neighbors')
plt.ylabel('scores')
plt.title('Performance of KNN')
plt.grid(linewidth=0.5)
plt.legend()

#Testing the model on the test data
knn_model = Pipeline([
        ('scaler', StandardScaler()),
        ('knn_clf', KNeighborsClassifier(n_neighbors=500,
                                metric='minkowski',
                                p=2))
    ])
knn_model.fit(X_selected_train, y_train)
y_pred_knn = knn_model.predict_proba(X_selected_test)[:,1]
plot_roc_graph(y_test, y_pred_knn, name='KNN (500)',
               color='black')

y_predictions_knn = model.predict(X_selected_test)
cm_knn = confusion_matrix(y_test, y_predictions)

scores_knn = compute_testscores(y_test, y_predictions_knn, y_pred_knn)

#%% Making custom subplot for knn and logistic
fig, axs = plt.subplots(2,1, figsize=(6,14))
fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred_logit)
roc_auc = metrics.auc(fpr, tpr)
axs[0].set_title('(1)',
              fontsize=22)
axs[0].plot(fpr, tpr, label='AUC {} = {}'.format('Logit' ,round(roc_auc, 3)),
            color='black', linewidth=1)
axs[0].plot([0,1], [0,1], 'r-', linewidth=1,
            label='benchmark')
axs[0].set_xlim([0,1])
axs[0].set_ylim([0,1])
axs[0].set_ylabel('True positive rate',
                  fontsize=20)
axs[0].set_xlabel('False positive rate',
                  fontsize=20)
axs[0].grid(linewidth=0.5)
axs[0].legend(fontsize=14)

fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred_knn)
roc_auc = metrics.auc(fpr, tpr)
axs[1].set_title('(2)',
              fontsize=22)
axs[1].plot(fpr, tpr, label='AUC {} = {}'.format('KNN (500)' ,round(roc_auc, 3)),
            color='black', linewidth=1)
axs[1].plot([0,1], [0,1], 'r-', linewidth=1,
            label='benchmark')
axs[1].set_xlim([0,1])
axs[1].set_ylim([0,1])
axs[1].set_ylabel('True positive rate',
                  fontsize=20)
axs[1].set_xlabel('False positive rate',
                  fontsize=20)
axs[1].grid(linewidth=0.5)
axs[1].legend(fontsize=14)

#%% Linear SVM
from sklearn.svm import SVC
regularisation = [10, 5, 1, 0.01, 0.001]
string_array_svm = [str(i) for i in regularisation] 

linear_SVM_scores = []
linear_SVM_scores_accuracy = []
linear_SVM_scores_f1 = []
for i in regularisation:
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('linear_svc', SVC(kernel='linear', 
                            C=i,
                            probability=True,
                            verbose=True))
        ])
    auc_score = compute_score(model, X_selected_train, y_train)
    linear_SVM_scores.append(auc_score)
    score_accuracy = compute_score_accuracy(model, X_selected_train, y_train)
    linear_SVM_scores_accuracy.append(score_accuracy)
    score_f1 = compute_score_f1(model, X_selected_train, y_train)
    linear_SVM_scores_f1.append(score_f1)

plt.plot(string_array_svm, linear_SVM_scores, 
         label='AUC', marker='o', linewidth=1.5,
         markersize=7)
plt.plot(string_array_svm, linear_SVM_scores_accuracy,
         label='accuracy', marker='o', linewidth=1.5,
         markersize=7)
plt.plot(string_array_svm, linear_SVM_scores_f1,
         label='f1', marker='o', linewidth=1.5,
         markersize=7)

plt.xlabel('C')
plt.ylabel('scores')
plt.title('Linear SVM classifier',
          fontsize=25)
plt.legend()

#Building linear SVM model with C=1
model_linear = Pipeline([
        ('scaler', StandardScaler()),
        ('linear_svc', SVC(kernel='linear', 
                            C=1,
                            probability=True,
                            verbose=True))
        ])
model_linear.fit(X_selected_train, y_train)
y_pred_linear = model_linear.predict_proba(X_selected_test)[:,1]
plot_roc_graph(y_test, y_pred_linear, name='SVM Linear',
               color='black')

y_predictions = model_linear.predict(X_selected_test)
cm_svm_linear = confusion_matrix(y_test, y_predictions)

scores_svm_linear = compute_testscores(y_test, y_predictions, y_pred_linear)

#%% Gaussian RBF SVM 
regularisation_gamma = [1, 0.1, 0.001, 0.0001, 0.00001]
regularisation_C = [10, 5, 1, 0.01, 0.001]
string_array_gamma = [str(i) for i in regularisation_gamma] 

#Vertical - gamma
#Horizontal - C
grid_scores_auc = np.zeros(shape=(len(regularisation_gamma),len(regularisation_C)))
grid_scores_acc = np.zeros(shape=(len(regularisation_gamma),len(regularisation_C)))
grid_scores_f1 = np.zeros(shape=(len(regularisation_gamma),len(regularisation_C)))

for gamma in range(len(regularisation_gamma)):
    for c in range(len(regularisation_C)):
        gaussian_model = Pipeline([
        ('scaler', StandardScaler()),
        ('svm_clf', SVC(kernel='rbf', 
                        gamma=regularisation_gamma[gamma], 
                        C=regularisation_C[c],
                        probability=True,
                        verbose=True))
        ])
        RBF_auc = compute_score(gaussian_model, X_selected_train, y_train)
        RBF_acc = compute_score_accuracy(gaussian_model, X_selected_train, y_train)
        RBF_f1 = compute_score_f1(gaussian_model, X_selected_train, y_train)
        grid_scores_auc[gamma, c] = RBF_auc
        grid_scores_acc[gamma, c] = RBF_acc
        grid_scores_f1[gamma, c] = RBF_f1

#Finding the index of maximum values
gamma_auc = regularisation_gamma[
            int(np.where(grid_scores_auc==np.max(grid_scores_auc))[0])
            ]
C_auc = regularisation_C[
        int(np.where(grid_scores_auc==np.max(grid_scores_auc))[1])
        ]

print('The model performs best in AUC measure with gamma={} and C={}, achieving AUC of {}'.format(
        gamma_auc, C_auc, np.max(grid_scores_auc)))
gamma_acc = regularisation_gamma[
            int(np.where(grid_scores_acc==np.max(grid_scores_acc))[0])
            ]
C_acc = regularisation_C[
        int(np.where(grid_scores_acc==np.max(grid_scores_acc))[1])
        ]
print('The model performs best in accuracy measure with gamma={} and C={}, achieving accuracy of {}'.format(
        gamma_acc, C_acc, np.max(grid_scores_acc)))
gamma_f1 = regularisation_gamma[
            int(np.where(grid_scores_f1==np.max(grid_scores_f1))[0])
            ]
C_f1 = regularisation_C[
        int(np.where(grid_scores_f1==np.max(grid_scores_f1))[1])
        ]
print('The model performs best in accuracy measure with gamma={} and C={}, achieving accuracy of {}'.format(
        gamma_f1, C_f1, np.max(grid_scores_f1)))

#Gamma = 0.001 is a clear winner in this regard, being preferred by all three models.
#In terms of the C parameter, in values between 0 and 10, it did not change much,
#Therefore we are going to use a standard C measure of 1
gaussian_model = Pipeline([
        ('scaler', StandardScaler()),
        ('svm_clf', SVC(kernel='rbf', 
                        gamma=0.001, 
                        C=10,
                        probability=True,
                        verbose=True))
        ])

gaussian_model.fit(X_selected_train, y_train)
y_pred_rbf = gaussian_model.predict_proba(X_selected_test)[:,1]
plot_roc_graph(y_test, y_pred_rbf, name='SVM Gaussian',
               color='black')

y_predictions = gaussian_model.predict(X_selected_test)
cm_svm_gaussian = confusion_matrix(y_test, y_predictions)

scores_svm_gaussian = compute_testscores(y_test, y_predictions, y_pred_rbf)

#%%
RBF_auc_scores = []
RBF_acc_scores = []
RBF_f1_scores = []
for i in regularisation_gamma:
    gaussian_model = Pipeline([
        ('scaler', StandardScaler()),
        ('svm_clf', SVC(kernel='rbf', 
                        gamma=i, 
                        C=5,
                        probability=True,
                        verbose=True))
        ])
    RBF_auc = compute_score(gaussian_model, X_selected_train, y_train)
    RBF_auc_scores.append(RBF_auc)
    RBF_acc = compute_score_accuracy(gaussian_model, X_selected_train, y_train)
    RBF_acc_scores.append(RBF_acc)
    RBF_f1 = compute_score_f1(gaussian_model, X_selected_train, y_train)
    RBF_f1_scores.append(RBF_f1)

plt.plot(string_array_gamma, RBF_auc_scores, 
         label='AUC', marker='o', linewidth=1.5,
         markersize=7)
plt.plot(string_array_gamma, RBF_acc_scores,
         label='accuracy', marker='o', linewidth=1.5,
         markersize=7)
plt.plot(string_array_gamma, RBF_f1_scores,
         label='f1', marker='o', linewidth=1.5,
         markersize=7)
plt.title('Gaussian RBF Kernel', 
          fontsize=25)
plt.xlabel(r'$\gamma$', 
           fontsize=12)
plt.ylabel('scores',
           fontsize=12)
plt.legend()

#Building linear SVM model with C=1
model = Pipeline([
        ('scaler', StandardScaler()),
        ('linear_svc', SVC(kernel='rbf', 
                            C=5,
                            probability=True,
                            verbose=True))
        ])
model.fit(X_selected_train, y_train)
y_pred = model.predict_proba(X_selected_test)[:,1]
plot_roc_graph(y_test, y_pred, name='SVM Gaussian',
               color='black')

y_predictions = model.predict(X_selected_test)
cm_svm_linear = confusion_matrix(y_test, y_predictions)

scores_svm_gaussian = compute_testscores(y_test, y_predictions, y_pred)
 
#%% Polynomial Kernel SVM 
degree = [2, 3, 4, 5, 10]
regularisation_C = [10, 5, 1, 0.01, 0.001]
string_array_degree = [str(i) for i in degree] 

#Vertical - degree
#Horizontal - error threshold
poly_scores_auc = np.zeros(shape=(len(degree),len(regularisation_C)))
poly_scores_acc = np.zeros(shape=(len(degree),len(regularisation_C)))
poly_scores_f1 = np.zeros(shape=(len(degree),len(regularisation_C)))

for i in range(len(degree)):
    for c in range(len(regularisation_C)):
        poly_model = Pipeline([
        ('scaler', StandardScaler()),
        ('svm_clf', SVC(kernel='poly', 
                        degree=degree[i], 
                        C=regularisation_C[c],
                        probability=True,
                        verbose=True))
        ])
        poly_auc = compute_score(poly_model, X_selected_train, y_train)
        poly_acc = compute_score_accuracy(poly_model, X_selected_train, y_train)
        poly_f1 = compute_score_f1(poly_model, X_selected_train, y_train)
        poly_scores_auc[i, c] = poly_auc
        poly_scores_acc[i, c] = poly_acc
        poly_scores_f1[i, c] = poly_f1

poly_degree_auc = degree[
            int(np.where(poly_scores_auc==np.max(poly_scores_auc))[0])
            ]
poly_C_auc = regularisation_C[
        int(np.where(poly_scores_auc==np.max(poly_scores_auc))[1])
        ]
print('The model performs best in AUC measure with degree={} and C={}, achieving AUC of {}'.format(
        poly_degree_auc, poly_C_auc, np.max(poly_scores_auc)))
#-------------------------------------------------------------------
poly_degree_acc = degree[
            int(np.where(poly_scores_acc==np.max(poly_scores_acc))[0])
            ]
poly_C_acc = regularisation_C[
        int(np.where(poly_scores_acc==np.max(poly_scores_acc))[1])
        ]
print('The model performs best in accuracy measure with degree={} and C={}, achieving accuracy of {}'.format(
        poly_degree_acc, poly_C_acc, np.max(poly_scores_acc)))
#-------------------------------------------------------------------
poly_degree_f1 = degree[
            int(np.where(poly_scores_f1==np.max(poly_scores_f1))[0])
            ]
poly_C_f1 = regularisation_C[
        int(np.where(poly_scores_f1==np.max(poly_scores_f1))[1])
        ]
print('The model performs best in accuracy measure with degree={} and C={}, achieving accuracy of {}'.format(
        poly_degree_f1, poly_C_f1, np.max(poly_scores_f1)))

#Once again we have a clear winner for degree (3), while in regards to error threshold
#examining closer the results, the best models perform also when the error threshold is
#left as C=1

#Building linear SVM model with C=1
model_poly = Pipeline([
        ('scaler', StandardScaler()),
        ('linear_svc', SVC(kernel='poly', 
                            C=10,
                            degree=3,
                            probability=True,
                            verbose=True))
        ])
model_poly.fit(X_selected_train, y_train)
y_pred_poly = model_poly.predict_proba(X_selected_test)[:,1]
plot_roc_graph(y_test, y_pred_poly, name='SVM Poly',
               color='black')

y_predictions = model_poly.predict(X_selected_test)
cm_svm_poly = confusion_matrix(y_test, y_predictions)

scores_svm_poly = compute_testscores(y_test, y_predictions, y_pred_poly)


#%%
poly_auc_scores = []
poly_acc_scores = []
poly_f1_scores = []

for i in degree:
    poly_model = Pipeline([
        ('scaler', StandardScaler()),
        ('svm_clf', SVC(kernel='poly', 
                        degree=i, 
                        C=0.1,
                        probability=True,
                        verbose=True))
        ])
    poly_auc = compute_score(poly_model, X_selected_train, y_train)
    poly_auc_scores.append(poly_auc)
    poly_acc = compute_score_accuracy(poly_model, X_selected_train, y_train)
    poly_acc_scores.append(poly_acc)
    poly_f1 = compute_score_f1(poly_model, X_selected_train, y_train)
    poly_f1_scores.append(poly_f1)

plt.plot(string_array_degree, poly_auc_scores, 
         label='AUC')
plt.plot(string_array_degree, poly_acc_scores,
         label='accuracy')
plt.plot(string_array_degree, poly_f1_scores,
         label='f1')
plt.legend()

#%% Plotting 3 roc curves
plt.figure(figsize=(6,6))
fpr_linear, tpr_linear, threshold_linear = metrics.roc_curve(y_test, y_pred_linear)
roc_auc_linear = metrics.auc(fpr_linear, tpr_linear)
plt.title('ROC curves for tuned models',
              fontsize=22)
plt.plot(fpr_linear, tpr_linear, label='AUC {} = {}'.format('linear' ,round(roc_auc_linear, 3)),
            color='black', linewidth=1)
plt.xlim([0,1])
plt.ylim([0,1])
plt.ylabel('True positive rate',
                  fontsize=20)
plt.xlabel('False positive rate',
                  fontsize=20)
fpr_logit, tpr_logit, threshold_logit = metrics.roc_curve(y_test, y_pred_logit)
roc_auc_logit = metrics.auc(fpr_logit, tpr_logit)
plt.plot(fpr_logit, tpr_logit, label='AUC {} = {}'.format('logit' ,round(roc_auc_logit, 3)), 
         linewidth=1, color='yellow')
fpr_poly, tpr_poly, threshold_poly = metrics.roc_curve(y_test, y_pred_poly)
roc_auc_poly = metrics.auc(fpr_poly, tpr_poly)
plt.plot(fpr_poly, tpr_poly, label='AUC {} = {}'.format('poly' ,round(roc_auc_poly, 3)), 
         linewidth=1, color='blue')
fpr_rbf, tpr_rbf, threshold_rbf = metrics.roc_curve(y_test, y_pred_rbf)
roc_auc_rbf = metrics.auc(fpr_rbf, tpr_rbf)
plt.plot(fpr_rbf, tpr_rbf, label='AUC {} = {}'.format('rbf' ,round(roc_auc_rbf, 3)), 
         linewidth=1, color='magenta')
fpr_knn, tpr_knn, threshold_knn = metrics.roc_curve(y_test, y_pred_knn)
roc_auc_knn = metrics.auc(fpr_knn, tpr_knn)
plt.plot(fpr_knn, tpr_knn, label='AUC {} = {}'.format('KNN (500)' ,round(roc_auc_knn, 3)), 
         linewidth=1, color='purple')
plt.plot([0,1], [0,1], 'r-', linewidth=1,
            label='benchmark')
plt.grid(linewidth=0.5)
plt.legend(fontsize=14)

#%% DECISION TREE
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(criterion='gini')

compute_score(model, X_selected_train, y_train)
compute_score_accuracy(model, X_selected_train, y_train)
compute_score_f1(model, X_selected_train, y_train)

model.fit(X_selected_train, y_train)
y_pred = model.predict(X_selected_test)
cm = confusion_matrix(y_test, y_pred)
#Tree pruning - using minimal cost complexity pruning, so we recursively find the weakest link
#(effective alpha), the nodes with the smallest effective alpha are pruned first.
#the function returns effective alphas and corresponding leaf impurities at each step
#As alpha ---> more of the tree is pruned, which increases the impurity of the leaves.
path = model.cost_complexity_pruning_path(X_selected_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

ccp_alphas, impurities = path.ccp_alphas, path.impurities
plt.figure(figsize=(10, 6))
plt.plot(ccp_alphas, impurities, 
         linewidth=1, color='black')
plt.xlabel("effective alpha")
plt.ylabel("total impurity of leaves")

clfs = []
for ccp_alpha in ccp_alphas:
    clf = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
    clf.fit(X_selected_train, y_train)
    clfs.append(clf)
tree_depths = [clf.tree_.max_depth for clf in clfs]
plt.figure(figsize=(10,  6))
plt.plot(ccp_alphas[:-1], tree_depths[:-1],
         linewidth=1, color='black')
plt.xlabel("effective alpha")
plt.ylabel("total depth")

acc_scores = [roc_auc_score(y_test, clf.predict(X_selected_test)) for clf in clfs]
tree_depths = [clf.tree_.max_depth for clf in clfs]
plt.figure(figsize=(10,  6))
plt.plot(ccp_alphas[:-1], acc_scores[:-1],
         linewidth=1, color='black')
plt.xlabel("effective alpha")
plt.ylabel("AUC")

maxval = max(acc_scores)
acc_scores.index(maxval)
maxalpha = ccp_alphas[151]

#Making new tree
tree_auc = []
tree_f1 = []
tree_acc = []
depth_vals = np.arange(2,50,1)
string_array_tree = [str(i) for i in depth_vals] 

for i in range(len(depth_vals)):
    model = DecisionTreeClassifier(criterion='gini',
                                   max_depth=depth_vals[i])
    auc = compute_score(model, X_selected_train, y_train)
    f1 = compute_score_f1(model, X_selected_train, y_train)
    acc = compute_score_accuracy(model, X_selected_train, y_train)
    tree_auc.append(auc)
    tree_f1.append(f1)
    tree_acc.append(acc)

plt.plot(string_array_tree, tree_auc, label='auc',
         linewidth=1.5)
plt.plot(string_array_tree, tree_f1, label='f1')
plt.plot(string_array_tree, tree_acc, label='accuracy')
plt.xlabel('Tree Depth')
plt.ylabel('score')
plt.legend()

#%% Bagging Classifier
from sklearn.ensemble import BaggingClassifier
estimators_arr = [1,10,100, 500]
string_array_estimators = [str(i) for i in estimators_arr] 
auc_scores_bag = []
acc_scores_bag = []
f1_scores_bag = []
for i in estimators_arr:
    model = BaggingClassifier(base_estimator=DecisionTreeClassifier(),
                              n_estimators=100, verbose=True)
    auc = compute_score(model, X_selected_train, y_train)
    auc_scores_bag.append(auc)
    acc = compute_score_accuracy(model, X_selected_train, y_train)
    acc_scores_bag.append(acc)
    f1 = compute_score_f1(model, X_selected_train, y_train)
    f1_scores_bag.append(f1)

plt.plot(string_array_estimators, auc_scores_bag, label='auc')
plt.plot(string_array_estimators, f1_scores_bag, label='f1')
plt.plot(string_array_estimators, acc_scores_bag, label='accuracy')
plt.xlabel('number of estimators')
plt.ylabel('score')
plt.legend()
#%% Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
rf_auc = []
rf_f1 = []
rf_acc = []
depth_vals = np.arange(2,30,1)
string_array_forest = [str(i) for i in depth_vals] 
for i in depth_vals:
    model = RandomForestClassifier(n_estimators=100,
                                   max_depth=i,
                                   verbose=True)
    auc_score = compute_score(model, X_selected_train, y_train)
    f1_score = compute_score_f1(model, X_selected_train, y_train)
    acc_score = compute_score_accuracy(model, X_selected_train, y_train)
    rf_auc.append(auc_score)
    rf_f1.append(f1_score)
    rf_acc.append(acc_score)
    
plt.title('Random Forest Classifier',
          fontsize=25)
plt.plot(string_array_forest, rf_auc[28:], label='auc',
         linewidth=1.5, marker='o', markersize=5)
plt.plot(string_array_forest, rf_f1[28:], label='f1',
         linewidth=1.5, marker='o', markersize=5)
plt.plot(string_array_forest, rf_acc[28:], label='accuracy',
         linewidth=1.5, marker='o', markersize=5)
plt.xlabel('Tree Depth')
plt.ylabel('score')
plt.legend() #A maximum depth of 5 appears to provide the best results

#%%
#Total Scores DF
final_df = pd.DataFrame(scores_logit,columns=['logit'])
final_df['KNN'] = pd.Series(scores_knn, index=final_df.index)
final_df['SVM L'] = pd.Series(scores_svm_linear, index=final_df.index)
final_df['SVM G'] = pd.Series(scores_svm_gaussian, index=final_df.index)
final_df['SVM P'] = pd.Series(scores_svm_poly, index=final_df.index)

fig, ax = plt.subplots(1,3, figsize=(18,6))
final_aucs = final_df.iloc[0,:]
clrs = ['grey' if (x < max(final_aucs)) else 'red' for x in final_aucs ]
sns.barplot(x=final_df.columns, y=final_aucs, 
            palette=clrs, ax=ax[0])
ax[0].set_ylim([0.71, 0.76])
ax[0].set_ylabel('AUC', 
                 fontsize=18)
ax[0].set_title('(1)',
                fontsize=22)

final_accs = final_df.iloc[1,:]
clrs = ['grey' if (x < max(final_accs)) else 'red' for x in final_accs]
sns.barplot(x=final_df.columns, y=final_accs,
            palette=clrs, ax=ax[1])
ax[1].set_ylim([0.64, 0.69])
ax[1].set_ylabel('accuracy', 
                 fontsize=18)
ax[1].set_title('(2)',
                fontsize=22)

final_f1s = final_df.iloc[2,:]
clrs = ['grey' if (x < max(final_f1s)) else 'red' for x in final_f1s]
sns.barplot(x=final_df.columns, y=final_f1s,
            palette=clrs, ax=ax[2])
ax[2].set_ylim([0.62, 0.69])
ax[2].set_ylabel('f1', 
                 fontsize=18)
ax[2].set_title('(3)',
                fontsize=22)
















