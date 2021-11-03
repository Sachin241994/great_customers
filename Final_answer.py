#!/usr/bin/env python
# coding: utf-8

# In[1]:


#all Importrant Libraries
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


# In[2]:


data=pd.read_csv("https://raw.githubusercontent.com/subashgandyer/datasets/main/great_customers.csv")


# In[3]:


data


# In[4]:


def cleaning_the_data(df):
    #Removing the Duplicates
    df.drop_duplicates(inplace=True)
    #Changing the Catogrical Variable
    df_dummie=pd.get_dummies(data[['workclass','marital-status','occupation','race','sex']])
    df=pd.concat([df_dummie,df],axis=1)
    df=df.drop(['workclass','marital-status','occupation','race','sex'],axis=1)
    #Filling the NA values
    df[['tea_per_year','coffee_per_year','mins_beerdrinking_year','mins_exercising_year']]=df[['tea_per_year','coffee_per_year','mins_beerdrinking_year','mins_exercising_year']].fillna(0)
    meanAge= data.groupby(['sex'])['age'].transform('mean')
    meansalary= data.groupby(['workclass','occupation'])['salary'].transform('mean')
    df['age'].fillna(value=meanAge, inplace=True)
    df['salary'].fillna(value=meansalary, inplace=True)
    df['salary']=df['salary'].fillna(0)
    return df


# In[5]:


df_clean=cleaning_the_data(data)
#Feature_Selection Start here
X=df_clean.drop(['great_customer_class'],axis=1)
y=df_clean['great_customer_class']
feature_name=list(X.columns)
num_feats=25


# In[6]:


def cor_selector(X, y,num_feats):
    
    cor_list = []
    feature_name = X.columns.tolist()
    for i in X.columns.tolist():
        cor = np.corrcoef(X[i], y)[0, 1]
        cor_list.append(cor)
    cor_list = [0 if np.isnan(i) else i for i in cor_list]
    cor_feature = X.iloc[:,np.argsort(np.abs(cor_list))[-num_feats:]].columns.tolist()
    cor_support = [True if i in cor_feature else False for i in feature_name]
    
    # Your code ends here
    return cor_support, cor_feature

def chi_squared_selector(X, y, num_feats):
    # Your code goes here (Multiple lines)
    X_norm = MinMaxScaler().fit_transform(X)
    chi_selector = SelectKBest(chi2, k=num_feats)
    chi_selector.fit(X_norm, y)
    chi_support = chi_selector.get_support()
    chi_feature = X.loc[:,chi_support].columns.tolist()
    # Your code ends here
    return chi_support, chi_feature

def rfe_selector(X, y, num_feats):
    # Your code goes here (Multiple lines)
    X_norm = MinMaxScaler().fit_transform(X)
    rfe_selector = RFE(estimator=LogisticRegression(), n_features_to_select=num_feats, step=10, verbose=5)
    rfe_selector.fit(X_norm, y)
    rfe_support = rfe_selector.get_support()
    rfe_feature = X.loc[:,rfe_support].columns.tolist()
    # Your code ends here
    return rfe_support, rfe_feature


def embedded_log_reg_selector(X, y, num_feats):
    # Your code goes here (Multiple lines)
    embedded_lr_selector = SelectFromModel(LogisticRegression(penalty="l2"), max_features=num_feats)
    embedded_lr_selector.fit(X, y)
    embedded_lr_support = embedded_lr_selector.get_support()
    embedded_lr_feature = X.loc[:,embedded_lr_support].columns.tolist()
    # Your code ends here
    return embedded_lr_support, embedded_lr_feature

def embedded_rf_selector(X, y, num_feats):
    # Your code goes here (Multiple lines)
    embedded_rf_selector = SelectFromModel(RandomForestClassifier(n_estimators=100), max_features=num_feats)
    embedded_rf_selector.fit(X, y)
    embedded_rf_support = embedded_rf_selector.get_support()
    embedded_rf_feature = X.loc[:,embedded_rf_support].columns.tolist()
    # Your code ends here
    return embedded_rf_support, embedded_rf_feature

def embedded_lgbm_selector(X, y, num_feats):
    lgbc=LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=32, colsample_bytree=0.2,
            reg_alpha=3, reg_lambda=1, min_split_gain=0.01, min_child_weight=40)

    embedded_lgbm_selector = SelectFromModel(lgbc, max_features=num_feats)
    embedded_lgbm_selector.fit(X, y)
    embedded_lgbm_support = embedded_lgbm_selector.get_support()
    embedded_lgbm_feature = X.loc[:,embedded_lgbm_support].columns.tolist()
    return embedded_lgbm_support, embedded_lgbm_feature

def preprocess_dataset(dataset_path):
    data = pd.read_csv("https://raw.githubusercontent.com/subashgandyer/datasets/main/great_customers.csv")
    data.drop_duplicates(inplace=True)
    df=pd.get_dummies(data[['workclass','marital-status','occupation','race','sex']])
    df=pd.concat([data,df],axis=1)
    df=df.drop(['workclass','marital-status','occupation','race','sex'],axis=1)
    df[['tea_per_year','coffee_per_year','mins_beerdrinking_year','mins_exercising_year']]=df[['tea_per_year','coffee_per_year','mins_beerdrinking_year','mins_exercising_year']].fillna(0)
    meanAge= data.groupby(['sex'])['age'].transform('mean')
    meansalary= data.groupby(['workclass','occupation'])['salary'].transform('mean')
    df['age'].fillna(value=meanAge, inplace=True)
    df['salary'].fillna(value=meansalary, inplace=True)
    df['salary']=df['salary'].fillna(0)
    features = df.columns
    X=df.drop(['great_customer_class'],axis=1)
    y=df['great_customer_class']
    num_feats=25
    
    return X, y, num_feats

def autoFeatureSelector(dataset_path, methods=[]):
    # Parameters
    # data - dataset to be analyzed (csv file)
    # methods - various feature selection methods we outlined before, use them all here (list)
    
    # preprocessing
    X, y, num_feats = preprocess_dataset(dataset_path)
    
    # Run every method we outlined above from the methods list and collect returned best features from every method
    if 'pearson' in methods:
        cor_support, cor_feature = cor_selector(X, y,num_feats)
    if 'chi-square' in methods:
        chi_support, chi_feature = chi_squared_selector(X, y,num_feats)
    if 'rfe' in methods:
        rfe_support, rfe_feature = rfe_selector(X, y,num_feats)
    if 'log-reg' in methods:
        embedded_lr_support, embedded_lr_feature = embedded_log_reg_selector(X, y, num_feats)
    if 'rf' in methods:
        embedded_rf_support, embedded_rf_feature = embedded_rf_selector(X, y, num_feats)
    if 'lgbm' in methods:
        embedded_lgbm_support, embedded_lgbm_feature = embedded_lgbm_selector(X, y, num_feats)
    
    
    # Combine all the above feature list and count the maximum set of features that got selected by all methods
    #### Your Code starts here (Multiple lines)
    feature_selection_df = pd.DataFrame({'Feature':feature_name, 'Pearson':cor_support, 'Chi-2':chi_support, 'RFE':rfe_support, 'Logistics':embedded_lr_support,
                                    'Random Forest':embedded_rf_support, 'LightGBM':embedded_lgbm_support})
    # count the selected times for each feature
    feature_selection_df['Total'] = np.sum(feature_selection_df, axis=1)
    # display the top 100
    feature_selection_df = feature_selection_df.sort_values(['Total','Feature'] , ascending=False)
    feature_selection_df.index = range(1, len(feature_selection_df)+1)
    best_features = feature_selection_df['Feature'].tolist()[:20]
    #### Your Code ends here
    return best_features


# In[7]:


cor_support, cor_feature = cor_selector(X, y,num_feats)
chi_support, chi_feature = chi_squared_selector(X, y,num_feats)
rfe_support, rfe_feature = rfe_selector(X, y,num_feats)
embedded_lr_support, embedded_lr_feature = embedded_log_reg_selector(X, y, num_feats)
embedder_rf_support, embedder_rf_feature = embedded_rf_selector(X, y, num_feats)
embedded_lgbm_support, embedded_lgbm_feature = embedded_lgbm_selector(X, y, num_feats)
best_features = autoFeatureSelector(dataset_path="https://raw.githubusercontent.com/subashgandyer/datasets/main/great_customers.csv", methods=['pearson', 'chi-square', 'rfe', 'log-reg', 'rf', 'lgbm'])


# In[8]:


#Data_Frame With final Selected feature
final_df=df_clean[best_features]


# In[9]:


# Feature and Classes starts here
X=final_df


# In[10]:


def prediction_models(X,y):
    RSEED=50
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20, random_state=42)
    lr = LogisticRegression()
    svm=SVC()
    rf=RandomForestClassifier(n_estimators=25,random_state=RSEED, max_features = 'sqrt',n_jobs=-1, verbose = 1)
    knn=KNeighborsClassifier(n_neighbors=3)
    gnb = GaussianNB()
    
    lr.fit(X,y)
    svm.fit(X,y)
    rf.fit(X,y)
    knn.fit(X,y)
    gnb.fit(X,y)
    
    pipeline = Pipeline([('model', lr)])
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    
    
    svm.predict(X_test)
    rf.predict(X_test)
    knn.predict(X_test)
    gnb.predict(X_test)
    
    
    lr_accuracy=round(np.mean(scores), 3)
    svm_accuracy=accuracy_score(y_test, svm.predict(X_test))
    rf_accuracy=accuracy_score(y_test, rf.predict(X_test))
    knn_accuracy=accuracy_score(y_test, knn.predict(X_test))
    gnb_accuracy=accuracy_score(y_test, gnb.predict(X_test))
    
    
    return lr_accuracy,svm_accuracy,rf_accuracy,knn_accuracy,gnb_accuracy


# In[11]:


def voting():
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20, random_state=42)
    model1 = LogisticRegression(random_state=1)
    model2 = DecisionTreeClassifier(random_state=1)
    model3 = KNeighborsClassifier()
    
    model = VotingClassifier(estimators=[('lr', model1), ('dt', model2), ('knn', model3)], voting='hard')
    model.fit(X_train, y_train)
    
    vote_accuracy=model.score(X_test, y_test)
    
    return vote_accuracy


# In[12]:


lr_accuracy,svm_accuracy,rf_accuracy,knn_accuracy,gnb_accuracy=prediction_models(X,y)
voting_accuracy=voting()


# In[13]:


print("Accuracy are as follow: lr_accuracy:",lr_accuracy,"svm_accuracy:",svm_accuracy,"rf_accuracy:",rf_accuracy,"knn_accuracy:",rf_accuracy,"gnb_accuracy:",gnb_accuracy,"and voting_accuracy:",voting_accuracy)


# In[ ]:





# In[ ]:




