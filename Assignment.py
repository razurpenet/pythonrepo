#!/usr/bin/env python
# coding: utf-8

# In[2]:


#work horses of tables (dataframe) and arrays
import pandas as pd
import numpy as np

# Machine Learning Algorithm

import matplotlib.pyplot as plot
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import sklearn.feature_selection
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# Test and train dataset split
from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings("ignore")
import io
import os
from sklearn.model_selection import cross_val_score
from sklearn import model_selection
from sklearn import tree
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score,classification_report, confusion_matrix
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()


# switch off pandas warning 
pd.options.mode.chained_assignment = None

# used to write the model to a file
from sklearn.externals import joblib


# In[4]:


#data is a dataframe 
#for undetectable missing values
missing_values = ["n/a", "?"," ."]

# Read the Donor Raw Data_ML with Python.csv file here
dataset = pd.read_csv('Donor Raw Data_ML with Python.csv', na_values = missing_values)


# In[5]:


dataset.sample(20)


# In[ ]:





# In[6]:


dataset.shape


# In[7]:


dataset.info()


# In[8]:


dataset.duplicated().sum()


# In[9]:


dataset.head(10)


# In[120]:


dataset.sample(10).transpose()


# In[12]:


dataset.columns


# In[ ]:





# In[13]:


dataset.isna().sum()


# In[14]:


dataset.describe()


# In[15]:


dataset.DONOR_AGE.mean()


# In[16]:


dataset.MONTHS_SINCE_LAST_PROM_RESP.mean()


# In[17]:


dataset.MONTHS_SINCE_LAST_PROM_RESP.fillna('19', inplace = True)


# In[20]:


median_age = dataset['DONOR_AGE'].mode()

print('median age', median_age)


# In[21]:


dataset.DONOR_AGE.fillna('67', inplace = True)


# In[22]:


mode_incomegroup = dataset['INCOME_GROUP'].mode()


print('Mode', mode_incomegroup)


# In[25]:


dataset.INCOME_GROUP.fillna('5', inplace = True)


# In[525]:


#dataset['DONOR_GENDER'].replace('F',0, inplace = True)
#dataset['DONOR_GENDER'].replace('M',1, inplace = True)


# In[26]:


#Replace " ? " with " U " - Unknown
dataset.URBANICITY.fillna('U', inplace = True)


# In[141]:


#check
dataset.groupby('SES').size()


# In[27]:


mode_ses = dataset['SES'].mode()
print('SES_mode', mode_ses)


# In[122]:


dataset.groupby('CLUSTER_CODE').size()


# In[126]:


dataset.CLUSTER_CODE.mode()


# In[142]:


dataset.CLUSTER_CODE.fillna(40, inplace = True)


# In[143]:


#1=yes, 0=no
dataset['TARGET_B'].replace(1,'Y', inplace = True)
dataset['TARGET_B'].replace(0,'N', inplace = True)


# In[144]:


dataset.columns


# In[146]:


gender_cat = pd.get_dummies(dataset, columns = ['DONOR_GENDER'], drop_first = True)
gender_cat.head(20).transpose()


# In[157]:


new_dataset = pd.get_dummies(dataset, columns = ['DONOR_GENDER',  'INCOME_GROUP'
                                                ,'URBANICITY', 'PUBLISHED_PHONE'
                                                ,'FREQUENCY_STATUS_97NK', 'HOME_OWNER'
                                                , 'IN_HOUSE', 'SES'], drop_first = True)


# In[158]:


new_dataset.head()


# In[159]:


new_dataset.shape


# In[160]:


new_dataset.columns.values


# In[162]:


feature_columns = new_dataset[['CONTROL_NUMBER', 'MONTHS_SINCE_ORIGIN',
       'DONOR_AGE', 'MOR_HIT_RATE',
       'MEDIAN_HOME_VALUE', 'MEDIAN_HOUSEHOLD_INCOME',
       'PCT_OWNER_OCCUPIED', 'PER_CAPITA_INCOME', 'PCT_ATTRIBUTE1',
       'PCT_ATTRIBUTE2', 'PCT_ATTRIBUTE3', 'PCT_ATTRIBUTE4', 'PEP_STAR',
       'RECENT_STAR_STATUS', 'RECENT_RESPONSE_PROP',
       'RECENT_AVG_GIFT_AMT', 'RECENT_CARD_RESPONSE_PROP',
       'RECENT_AVG_CARD_GIFT_AMT', 'RECENT_RESPONSE_COUNT',
       'RECENT_CARD_RESPONSE_COUNT', 'MONTHS_SINCE_LAST_PROM_RESP',
       'LIFETIME_CARD_PROM', 'LIFETIME_PROM', 'LIFETIME_GIFT_AMOUNT',
       'LIFETIME_GIFT_COUNT', 'LIFETIME_AVG_GIFT_AMT', 'LIFETIME_GIFT_RANGE',
       'LIFETIME_MAX_GIFT_AMT', 'LIFETIME_MIN_GIFT_AMT', 'LAST_GIFT_AMT',
       'CARD_PROM_12', 'NUMBER_PROM_12', 'MONTHS_SINCE_LAST_GIFT',
       'MONTHS_SINCE_FIRST_GIFT', 'FILE_AVG_GIFT', 'FILE_CARD_GIFT',
       'DONOR_GENDER_F', 'DONOR_GENDER_M', 'DONOR_GENDER_U',
       'INCOME_GROUP_2.0', 'INCOME_GROUP_3.0', 'INCOME_GROUP_4.0',
       'INCOME_GROUP_5.0', 'INCOME_GROUP_6.0', 'INCOME_GROUP_7.0',
       'INCOME_GROUP_5', 'URBANICITY_R', 'URBANICITY_S', 'URBANICITY_T',
       'URBANICITY_U', 'PUBLISHED_PHONE_1', 'FREQUENCY_STATUS_97NK_2',
       'FREQUENCY_STATUS_97NK_3', 'FREQUENCY_STATUS_97NK_4', 'HOME_OWNER_U',
       'IN_HOUSE_1','SES_2.0', 'SES_3.0', 'SES_4.0', 'SES_5.0']]


# In[163]:


feature_columns.head().transpose()


# In[164]:


cn_col = dataset[['CONTROL_NUMBER']]
cn_col.head()


# In[ ]:





# In[165]:


feature_columns.isnull().sum()


# In[166]:


target_col = ['TARGET_B']
target = dataset[target_col]
target.head()


# In[167]:


sns.countplot(x='TARGET_B',data=dataset);


# In[43]:


sns.factorplot(x='TARGET_B', col='FREQUENCY_STATUS_97NK_4', kind='count', data=new_dataset ).set_xticklabels(rotation=0);


# In[168]:


K_Best=sklearn.feature_selection.SelectKBest(k=8)
selected_features=K_Best.fit(feature_columns,target)
indices_selected=selected_features.get_support(indices=True)
chosen_cols=[feature_columns.columns[i] for i in indices_selected]


# In[169]:


chosen_cols


# In[170]:


X = new_dataset[chosen_cols]


# In[171]:


X.sample(3)


# In[172]:


y = target['TARGET_B']


# In[173]:


y.head()


# In[266]:


from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=72)


# In[191]:


X_train.shape


# In[192]:


X_test.shape


# In[193]:


y_train.shape


# In[194]:


y_test.shape


# In[285]:


#modelling with random forest

randForest = RandomForestClassifier(n_estimators=50, min_samples_split=50, max_depth=7, 
                                    max_features=1)


# In[286]:


randForest.fit(X_train,y_train.values)


# In[287]:


y_pred_RF  = randForest.predict(X_test)


# In[288]:


randForestScore = accuracy_score(y_test,y_pred_RF)


# In[289]:


print("The Random forest accuraccy score is:", randForestScore)


# In[239]:


pd.crosstab(y_test, y_pred_RF, rownames = ['Actual'], colnames = ['Predicted'])


# In[189]:


# Testing Accuracy
print(accuracy_score(y_test, y_pred_RF))
print(confusion_matrix(y_test, y_pred_RF))
print(classification_report(y_test, y_pred_RF))


# In[62]:


conf_m = confusion_matrix(y_test, y_pred_RF)
cf_mat_p = conf_m/conf_m.sum()

plt.figure(figsize=(12,8))
sns.heatmap(cf_mat_p, annot=True, linewidths=.5, cmap=cm.summer,xticklabels=['No','Yes'], yticklabels=['No','Yes'])
plt.ylabel('Actual')
plt.xlabel('Predicted label')
plt.title("Normalized Confusion Matrix")
plt.show()


# In[278]:


#modellimg with logistic regression

from sklearn.linear_model import LogisticRegression


# In[279]:


logReg = LogisticRegression()


# In[280]:


logReg.fit(X_train,y_train.values)


# In[281]:


logReg_predict =logReg.predict(X_test)


# In[282]:


# Predicted Probabilities
probabilities = logReg.predict_proba(X_test)


# In[283]:


# show the first 3 rows
probabilities[:3]


# In[293]:


accuracy_score(y_test,logReg_predict)


# In[70]:


pd.crosstab(y_test,logReg_predict, rownames = ['Actual'], colnames = ['Predicted'])


# In[72]:


print(accuracy_score(y_test,logReg_predict))
print(confusion_matrix(y_test,logReg_predict))
print(classification_report(y_test,logReg_predict))


# In[84]:


#modelling with KneighborsClassifier

from sklearn.neighbors import KNeighborsClassifier


# In[85]:


knn = KNeighborsClassifier(n_neighbors =5)


# In[86]:


knn.fit(X_train,y_train.values)


# In[87]:


KNN_pred = knn.predict(X_test)


# In[88]:


accuracy_score(y_test,KNN_pred)


# In[89]:


pd.crosstab(y_test,KNN_pred, rownames = ['Actual'], colnames = ['Predicted'])


# In[90]:


print(accuracy_score(y_test,KNN_pred))
print(confusion_matrix(y_test,KNN_pred))
print(classification_report(y_test,KNN_pred))


# In[93]:


#modelling with Decision Tree

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier


# In[94]:


DecTree = tree.DecisionTreeClassifier(max_depth=10,max_features=4 )


# In[95]:


DecTree.fit(X_train,y_train)


# In[96]:


DecTree_pred = DecTree.predict(X_test)


# In[97]:


accuracy_score(y_test,DecTree_pred)


# In[98]:


pd.crosstab(y_test,DecTree_pred, rownames = ['Actual'], colnames = ['Predicted'])


# In[294]:


print('KNN : ',accuracy_score(y_test,KNN_pred)  )
print('Log_Reg: ', accuracy_score(y_test,logReg_predict))
print('R_Forest: ',accuracy_score(y_test,y_pred_RF))
print('Decision_Forest: ', accuracy_score(y_test,DecTree_pred))


# In[321]:


# CossValidation

validation_size = 0.20
seed = 9
scoring = 'accuracy'


# In[322]:


X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, y, test_size=validation_size, 
                                                                                random_state=seed,stratify=y)


# In[323]:


models = []
models.append(('LR_cv', LogisticRegression()))
models.append(('KNN_cv', KNeighborsClassifier()))
models.append(('RF_cv', RandomForestClassifier()))
models.append(('DT_cv', DecisionTreeClassifier()))


# In[324]:


results = []
names = []
for name, model in models:
       
    kfold = model_selection.KFold(n_splits=10, random_state=seed)

    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)


# In[306]:


LR_cv = LogisticRegression()
LR_cv.fit(X_train, Y_train)

LR_cv_pred = LR_cv.predict(X_validation)
print(accuracy_score(Y_validation, LR_cv_pred))
print(confusion_matrix(Y_validation, LR_cv_pred))
print(classification_report(Y_validation, LR_cv_pred))


# In[326]:


print( LR_cv.predict_proba(X_validation))


# In[325]:


pd.crosstab(Y_validation,LR_cv_pred, rownames = ['Actual'], colnames = ['Predicted'])


# In[327]:


conf_m = confusion_matrix(Y_validation, LR_cv_pred)
cf_mat_p = conf_m/conf_m.sum()

plt.figure(figsize=(12,8))
sns.heatmap(cf_mat_p, annot=True, linewidths=.5, cmap=cm.summer,xticklabels=['No','Yes'], yticklabels=['No','Yes'])
plt.ylabel('Actual')
plt.xlabel('Predicted label')
plt.title("Normalized Confusion Matrix")
plt.show()


# In[328]:


#Using best Modelto Predict Prospective Donor

#for undetectable missing values
missing_values = ["n/a", "?"," ."]

# Read the Donor Raw Data_ML with Python.csv file here
pros_data = pd.read_csv('Prospective Donor_ML with Python.csv', na_values = missing_values)


# In[329]:


pros_data.shape


# In[346]:


pros_data.head(2).transpose()


# In[331]:


pros_data.isnull().sum()


# In[332]:


pros_data.DONOR_AGE.median()


# In[333]:


pros_data.groupby('SES').size()


# In[ ]:


pros_data.DONOR_AGE.fillna(59, inplace = True)
pros_data.URBANICITY.fillna('U', inplace = True)
pros_data.URBANICITY.fillna(5.0, inplace = True)
pros_data.CLUSTER_CODE.fillna(pros_data.CLUSTER_CODE.mode(), inplace = True)
pros_data.INCOME_GROUP.fillna(pros_data.INCOME_GROUP.mode(), inplace = True)


# In[347]:


#pros_data.drop(['WEALTH_RATING', 'TARGET_D'], axis = 1, inplace = True)


# In[348]:


pros_datanew = pd.get_dummies(dataset, columns = ['DONOR_GENDER',  'INCOME_GROUP'
                                                ,'URBANICITY', 'PUBLISHED_PHONE'
                                                ,'FREQUENCY_STATUS_97NK', 'HOME_OWNER'
                                                , 'IN_HOUSE', 'SES'], drop_first = True)


# In[349]:


pros_datanew.head(4)


# In[338]:


pros_datanew.columns


# In[351]:


feature_columns = ['PEP_STAR',
 'RECENT_RESPONSE_PROP',
 'RECENT_CARD_RESPONSE_PROP',
 'RECENT_RESPONSE_COUNT',
 'RECENT_CARD_RESPONSE_COUNT',
 'LIFETIME_GIFT_COUNT',
 'FILE_CARD_GIFT',
 'FREQUENCY_STATUS_97NK_4']


# In[352]:


feature_columns


# In[354]:


new_app = pros_datanew[feature_columns]


# In[355]:


new_app.head(3)


# In[378]:


New_App_Score = LR_cv.predict(new_app)


# In[380]:


New_App_Score[:200]


# In[369]:


##Store the final result in a csv file to be passed to decision makers to use.

pd.DataFrame({'CONTROL_NUMBER':pros_datanew.CONTROL_NUMBER,'TARGET_B':New_App_Score}).to_csv('Result.csv',index=False)


# In[371]:


Ne = pd.DataFrame({'CONTROL_NUMBER':pros_datanew.CONTROL_NUMBER,'TARGET_B':New_App_Score})


# In[377]:


Ne

