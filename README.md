# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Upload the file to your cell.
2.Type the required program.
3.Print the program.
4.End the program. 
 


## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: k.kavya
RegisterNumber: 22008613 
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error , mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()

df.tail()

X = df.iloc[:,:-1].values
X

y = df.iloc[:,-1].values
y

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)
y_pred=regressor.predict(X_test)

y_pred

y_test

plt.scatter(X_train,y_train,color="green")
plt.plot(X_train,regressor.predict(X_train),color="blue")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(X_test,y_test,color="grey")
plt.plot(X_test,regressor.predict(X_test),color="purple")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```

## Output:

![image](https://user-images.githubusercontent.com/118668727/229502234-67d8bf45-0f64-4704-8d4b-3663caa4ad9e.png)
![image](https://user-images.githubusercontent.com/118668727/229505603-298bb330-a7b2-4caa-b0dc-960343573dae.png)
![image](https://user-images.githubusercontent.com/118668727/229505902-026aa2cb-319c-4e4c-bade-dc13bdd0d503.png)
![image](https://user-images.githubusercontent.com/118668727/229506020-ed6533c6-ff7a-4ed4-b860-21b43f8d3d8a.png)
![image](https://user-images.githubusercontent.com/118668727/229506104-bfa28d21-2de9-47fb-b866-ea3cd147c519.png)
![image](https://user-images.githubusercontent.com/118668727/229506452-47ad76f6-c1ce-4318-bb60-62c23d08f55f.png)
![image](https://user-images.githubusercontent.com/118668727/229506537-7ab241c6-650f-4cdc-b29d-b94e27caf2c5.png)
![image](https://user-images.githubusercontent.com/118668727/229506626-2707d72e-2a52-4b34-8d8f-ab64b10568f6.png)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
