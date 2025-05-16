# -*- coding: utf-8 -*-
"""
Created on Fri May 16 19:34:30 2025

@author: SOGUTPC
"""
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from statsmodels.formula.api import ols

#Load dataset
penguins = pd.read_csv('penguins.csv')
penguins.head()

#subset data
penguins = penguins[['body_mass_g','bill_length_mm','sex','species']]
#rename columns
penguins.columns=['body_mass_g','bill_length_mm','gender','species']
#drop missing values
penguins.dropna(inplace=True)
#reset index
penguins = penguins.reset_index(inplace=True,drop=True)

#select X and Y variables
penguins_x = penguins[['bill_length_mm','gender','species']]
penguins_y = penguins[['body_mass_g']]

#creating train and test datasets
X_train,X_test,y_train,y_test = train_test_split(penguins_x,penguins_y,test_size=0.3,random_state=42)

#writing ols formula as a string
ols_formula = 'body_mass_g ~ bill_length_mm + C(gender) + C(species)'

#create a ols dataframe
ols_data = pd.concat([X_train,y_train],axis=1)
#creating ols object and fitting model
OLS = ols(ols_formula,ols_data)
model=OLS.fit()
#showing predictions
predictions = model.predict(X_test)
print(predictions.head())

#model results
model.summary()


