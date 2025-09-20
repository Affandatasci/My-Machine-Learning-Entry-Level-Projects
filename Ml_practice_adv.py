from os.path import split

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from altair.utils.save import set_inspect_format_argument
from pandas.core.algorithms import value_counts
from pandas.core.common import random_state
from pandas.io.sas.sas_constants import column_format_text_subheader_index_offset, subheader_count_length

from sklearn.neighbors import KNeighborsClassifier


from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_absolute_error
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.metrics import classification_report, confusion_matrix


df=sns.load_dataset('iris')
df=pd.DataFrame(df)

# print(df.columns)

x=df.drop(columns='species')
y=df.iloc[:,-1]
# print(y)

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,
    random_state=42)

# fruits={
#     'halal':'aam',
#     'haram':'hara aam'
# }
# print(fruits)
#
models={
    'knn':{
        'model':KNeighborsClassifier(),
        'params':{
            'n_neighbors':[2,5,10],
            'weights':['uniform','distance']
        }
    },
    'SVM':{
        'model':SVC(),
        'params':{
            'C':[0.1,1,10],
            'kernel':['linear','rbf','poly']
        }
    },
'RandomForest': {
        'model': RandomForestClassifier(),
        'params': {
            'n_estimators': [50, 100, 150],
            'max_depth': [None, 10, 20]
}
}}
best_model={}
best_score=0
best_model_name=None

for model,config in models.items():

     grid_search=GridSearchCV(estimator=config['model'],param_grid=config['params'],cv=5,
                              scoring='accuracy')
     grid_search.fit(x_train,y_train)

     best_model[model]={
         'best_params':grid_search.best_params_,
         'best_score':grid_search.best_score_
     }
     y_pred=grid_search.predict(x_test)
     test_score=accuracy_score(y_test,y_pred)

     if test_score>best_score:
         best_score=test_score
         best_model_name=model

print(best_model_name)
























