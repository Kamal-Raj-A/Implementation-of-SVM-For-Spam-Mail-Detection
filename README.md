# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages.
2. Import the dataset to operate on.
3. Split the dataset.
4. Predict the required output.
5. End the program.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: H VISHINU
RegisterNumber:  212223220124
*/
```
```
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
df=pd.read_csv("letter-recognition.csv")
df.head()
df.isnull().sum()
df.info()
x=df.iloc[:,1:].values
y=df.iloc[:,0].values
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)
model=SVC()
model
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
y_pred
score=accuracy_score(y_test,y_pred)
print(score)
```

## Output:
### Data Head:

![Screenshot 2024-05-07 083418](https://github.com/VisHinu24/Implementation-of-SVM-For-Spam-Mail-Detection/assets/144244396/29117440-d3ff-413c-bbb0-be046677a57f)

### Data Info:
![Screenshot 2024-05-07 083437](https://github.com/VisHinu24/Implementation-of-SVM-For-Spam-Mail-Detection/assets/144244396/762c7c9c-4ff4-48a5-a58c-13b090b4dccd)


### Data isnull():

![Screenshot 2024-05-07 083428](https://github.com/VisHinu24/Implementation-of-SVM-For-Spam-Mail-Detection/assets/144244396/bfb1c7c7-0383-4a8c-924e-c7221ac0a958)

### y_pred:
![Screenshot 2024-05-07 083459](https://github.com/VisHinu24/Implementation-of-SVM-For-Spam-Mail-Detection/assets/144244396/0c0f4eac-6e2e-4a93-b638-b518b982f7b6)


### Accuracy:

![Screenshot 2024-05-07 083503](https://github.com/VisHinu24/Implementation-of-SVM-For-Spam-Mail-Detection/assets/144244396/9ca8cbbe-fa82-43f2-8356-7ba97baa654b)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
