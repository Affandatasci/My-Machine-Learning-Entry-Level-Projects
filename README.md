# My-Machine-Learning-Entry-Level-Projects
from os.path import split

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from altair.utils.save import set_inspect_format_argument
from pandas.core.algorithms import value_counts
from pandas.core.common import random_state
from pandas.io.sas.sas_constants import column_format_text_subheader_index_offset, subheader_count_length

from sklearn.impute import SimpleImputer

from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_absolute_error
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix




df=pd.read_csv(r'C:\Users\Affan Ahmed Khan\Downloads\archive\winequality-red.csv')
# after uploading the files, we always check its features
# df.shape
# df.describe()
# df.head(5)
# df.tail()
# df.info()
# df.columns

# df.isnull().sum()
# df.duplicated().sum()

# df.value_counts('alcohol')
# df.value_counts('quality')

# print(df.isnull().sum())
# imputer=SimpleImputer(strategy='median')
# df[df.columns]=imputer.fit_transform(df)

#DROpping duplicates

# print(df.duplicated().sum())
# df=df.drop_duplicates()
# print(df.duplicated().sum())


# cv=df.value_counts('quality')
# print(cv)

# plt.figure(figsize=(15,8))
# sns.countplot(df,x='quality')
# plt.show()

# sns.heatmap(df.corr())
# plt.show()


# Separate features and target
# x = df.drop('quality', axis=1)

x=df.drop(columns=['quality'])
y=df['quality']
# # print(y)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

scaler=StandardScaler()
x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.fit_transform(x_test)
#
# # print(x_train_scaled.head()) #give error
#
x_train=pd.DataFrame(x_train_scaled,columns=x.columns)
x_test=pd.DataFrame(x_test_scaled,columns=x.columns)

# print(x_train.head())

# model=RandomForestRegressor(random_state=42)
# print(model.get_params())

param_grid={
     'n_estimators':[None,100,200],
     'max_samples':[None,10,20],
      'min_samples_leaf':[None,5,10]
}
model=GridSearchCV(RandomForestRegressor(random_state=42),param_grid=param_grid,cv=5,
                   scoring='neg_mean_squared_error',n_jobs=-1)

model.fit(x_train,y_train)
best_rf=model.best_estimator_
print(model.best_params_)

# Scooring on training data

score=cross_val_score(best_rf,x_train,y_train,cv=5,scoring='neg_mean_squared_error')
print(score.mean())

# Scooring on training data

y_pred=best_rf.predict(x_test)

mse=mean_squared_error(y_test,y_pred)
mae=mean_absolute_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)

print(mse,mae,r2)


# Plot Y_Actual vs predicted_Y
# plt.figure(figsize=(10,7))
# plt.scatter(y_test,y_pred,alpha=0.5)
# plt.plot([y.min(),y.max()],[y.min(),y.max()],'r--')
# plt.xlabel('Actual Quality')
# plt.ylabel('Pre Quality')
# plt.title('Actual vs Predicted Wine Quality')
# plt.show()


# Feature Importance
# importances = pd.Series(best_rf.feature_importances_, index=x.columns).sort_values(ascending=False)
# plt.figure(figsize=(10, 6))
# sns.barplot(x=importances.values, y=importances.index, palette='coolwarm')
# plt.title('Feature Importances from Random Forest')
# plt.show()


# importances=pd.Series(best_rf.feature_importances_,index=x.columns).sort_values(ascending=False)
# print(importances)
#
# sns.barplot(importances)
# plt.show()

import joblib
joblib.dump(best_rf, 'wine_quality_rf_model.pkl')
print("Model saved as 'wine_quality_rf_model.pkl'")

# Load
loaded_model = joblib.load('wine_quality_rf_model.pkl')
































