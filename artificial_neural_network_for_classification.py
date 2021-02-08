# -*- coding: utf-8 -*-

### Importing the libraries
"""

#Google Colab already comes installed with tensorflow but we must import it
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#To check if tensorflow library is loaded
tf.__version__

"""## Part 1 - Data Preprocessing

### Importing the dataset
"""

dataset=pd.read_csv('Churn_Modelling.csv')
#Removing the unwanted columns.Same operation can done using drop() method.
X=dataset.iloc[:,3:-1].values
Y=dataset.iloc[:,-1].values

#Displaying the Features or X
print(X)

#Displaying the target label or Y
print(y)

"""### Encoding categorical data"""

from sklearn.preprocessing import LabelEncoder
#Using LabelEncoder() for gender as it has 2 categories
le=LabelEncoder()
X[:,2]=le.fit_transform(X[:,2])

"""Label Encoding the "Gender" column"""

print(X[1])

"""One Hot Encoding the "Geography" column"""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
#Using OneHotEncoder() for country as it has more than 2 categories
ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[1])],remainder='passthrough')
X=np.array(ct.fit_transform(X))

print(X[1])

"""### Splitting the dataset into the Training set and Test set"""

from sklearn.model_selection import train_test_split
#Splitiing the train and test data using sklearn library predefined method train_test_split()
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

"""### Feature Scaling"""

from sklearn.preprocessing import StandardScaler
#1.Scaling the x_train and x_test data.
#2.fit_transform must be used first while scaling to fit the appropriate data and then transform() can be used to compute based on the same fitted data
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

"""## Part 2 - Building the ANN

### Initializing the ANN
"""

#creating the instance of the artificial neural network using keras.Sequential()
ann=tf.keras.models.Sequential()

"""### Adding the input layer and the first hidden layer"""

#Creating the input layer and first hidden layer of the Aritificial Neural Network
#units : Number of neurons in the layer, 6 neurons are created here
#activation : activation function used , 'relu' --> rectifier function
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

"""### Adding the second hidden layer"""

ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

"""### Adding the output layer"""

#'sigmoid' activation function is used as it gives the probability also
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

"""## Part 3 - Training the ANN

### Compiling the ANN
"""

#optimizer: function used to optimise the NN using backpropogation, 'adam' refers to stochiastic gradient descent
#loss: function used to calculate loss function, 'binary_crossentropy' is used as there are only 2 possible outcomes 
ann.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

"""### Training the ANN on the Training set"""

#batch_size: Number of rows that should be taken as a batch to compute and optimise the neural network
#epochs: Number of cycles(going through entire train dataset) the model should undergo
ann.fit(x_train,y_train,batch_size=32,epochs=111)

"""## Part 4 - Making the predictions and evaluating the model

### Predicting the result of a single observation

Using our ANN model to predict if the customer with the following informations will leave the bank: 

Geography: France

Credit Score: 600

Gender: Male

Age: 40 years old

Tenure: 3 years

Balance: \$ 60000

Number of Products: 2

Does this customer have a credit card? Yes

Is this customer an Active Member: Yes

Estimated Salary: \$ 50000

So, should we say goodbye to that customer?
"""

#Passing the data to predict as 2-D array is necessary.Above details are converted to model understandable data as passed through predict() method.
print(ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])) > 0.5)

"""**Solution**

Therefore, our ANN model predicts that this customer stays in the bank!

**Important note 1:** Notice that the values of the features were all input in a double pair of square brackets. That's because the "predict" method always expects a 2D array as the format of its inputs. And putting our values into a double pair of square brackets makes the input exactly a 2D array.

**Important note 2:** Notice also that the "France" country was not input as a string in the last column but as "1, 0, 0" in the first three columns. That's because of course the predict method expects the one-hot-encoded values of the state, and as we see in the first row of the matrix of features X, "France" was encoded as "1, 0, 0". And be careful to include these values in the first three columns, because the dummy variables are always created in the first columns.

### Predicting the Test set results
"""

#Using the predict() for the test data and converting them into yes/no category based on their probability
y_pred=ann.predict(x_test)
y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))

"""### Making the Confusion Matrix"""

#Building the confusion matrix from predefined method from sklearn library
from sklearn.metrics import confusion_matrix,accuracy_score
cm=confusion_matrix(y_test,y_pred)
print(cm)
accuracy_score(y_test,y_pred)