#!/usr/bin/env python
# coding: utf-8

# In[4]:


# Read the Titanic_train.csv file here
#data is a dataframe 
data = pd.read_csv('titanic_train.csv')


# In[5]:


# shape
data.shape


# In[6]:


# tail
data.tail(10)


# In[7]:


# sample
data.sample(10)


# In[8]:


# specific lines
data.iloc[500:501]


# In[ ]:


# columns


# In[9]:


data.groupby('pclass').size()


# In[10]:


data.isna().sum()


# In[10]:


# Statistics summary
data.describe


# In[11]:


# here I'm dropping out all lines where the target variable is missing
# axis = 0 means deleting rows
# how = 'any' deleting when any value on that row is missing
# subset the column you want to test for missing values
# inplace = true means we modify the data table ///so be careful
data.dropna(axis = 0, how ='any',subset=['survived'],inplace= True)

# alternative to create a new dataset
new_dataset = data.dropna(axis = 0, how ='any',subset=['survived'],inplace= False)

# axis =0 is dropping rows
# how =any is dropping when there is an 'NaN' is the subset
# subset = list of columns to consider


# For this example, we are only extracting 2 things: Class, age and sex. 
# Do that below
# 
# The best place to deal with data and missing data is in the dataframe BEFORE it becomes an array

# In[12]:


# Extract the pclass, age and sex into a new Dataframe
column_list = ['pclass','age','sex', 'survived']
# reduced dataset
titanic_data = data[column_list]

# exactly the same titanic_data = data[['pclass','age','sex']]
print(titanic_data.head())


# In[13]:


# Convert pclass to pure numbers
# inplace = True means i'm modifying the dataframe titanic_data

titanic_data['pclass'].replace('1st',1,inplace = True)
titanic_data['pclass'].replace('2nd',2,inplace = True)
titanic_data['pclass'].replace('3rd',3,inplace = True)

print(titanic_data.tail(15))
# a parte comment
# inplace = False means I need to assign that dataframe
# new_data =titanic_data['pclass'].replace('1st',1,inplace = False)


# In[ ]:


# Replace the sex with 0 for female, 1 for male
# syntax is column in dataframe = np.where (condition / if statement involving
# a column comma, if yes comma else)


# In[ ]:





# In[ ]:


print(titanic_data.head(15))


# In[12]:


# deal with age missing values

# 1st step calculate the median for the whole population
median_age = titanic_data['age'].median()
print('median',median_age)


# In[1]:


# mode not useful for quantitative  data
# beware that mode returns the most frequent categories.
# there can be 2 or more figures
# to pick up the first mode_age[0]
mode_class = titanic_data['pclass'].mode().mean()
print('mode',mode_class)


# In[ ]:


# descriptive functions google pandas dataframe descriptive stats

# 2nd step to replace NaN by the median

titanic_data['age'].fillna(median_age, inplace = True)

# equivalent to
# titanic_data['age'].replace(np.nan, median_age, inplace = True)
print(titanic_data.head(10))


# In[ ]:


# Create the expected result dataframe
survived_d = titanic_data['survived']

print('survivors in our sample',survived_d.sum())
print('missing values in target', survived_d.isna().sum())
#print(survived_d)

### we want zero missing values in the target variable


# In[ ]:


# Create train/test split

# the percentage of the dataset in the test set
test_percentage = 0.20 #20%

# the random seed a we are sampling randomly our data
# that is we select the lines in the 2 different set randomly
seed = 97

# Split-out validation dataset


X = titanic_data[['pclass', 'age', 'sex']]
Y = survived_d
X_train, X_test, Y_train, Y_test =  model_selection.train_test_split(X, Y, test_size=test_percentage, random_state=seed)


# we now have X_train and Y_train for preparing models
# and  X_validation and Y_validation we can use later

print(len(X_train), len(Y_train))


# In[ ]:


# Create the random forest instance, and train it with training data

rf = RandomForestClassifier (n_estimators = 100)

rf.fit(X_train,Y_train.values.ravel())


# In[ ]:


# Get the accuracy of your model

accuracy = rf.score(X_test, Y_test)
print('My first model accuracy is = {}%'.format(accuracy*100))


# In[ ]:


# this is for random forest

importances = rf.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]
print(X.columns[indices[1]])
# Print the feature ranking
print("Feature ranking:")

for f in range(X.shape[1]):
    #print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
    print( f + 1, X.columns[indices[f]], importances[indices[f]])


# 

# In[ ]:





# In[ ]:


# Write the model to a file called "titanic_model1"


# In[ ]:


joblib.dump(rf,'titanic_model1',compress =0)


# In[ ]:


loaded_model = joblib.load('titanic_model1')
result = loaded_model.score(X_test,Y_test)
print(result)

