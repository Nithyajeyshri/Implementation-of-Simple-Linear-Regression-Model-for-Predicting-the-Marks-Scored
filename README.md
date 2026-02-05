# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Collect the dataset of input and output.

2.Load and preprocess the data.

3.Train the Simple Linear Regression model using the training data.

4.Use the trained model to predict the marks for new input values.
## Program:
```
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load data
df = pd.read_csv("/content/score_updated.csv")
display(df.head(10))

# Visualize data
plt.scatter(df['Hours'], df['Scores'])
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.title('Hours vs Scores')
plt.show()

x = df.iloc[:, 0:1]
y = df.iloc[:, -1]

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Train model
lr = LinearRegression()
lr.fit(X_train, Y_train)

# Correcting variable case and generating predictions
print("X_train samples:", X_train.head())
print("Y_train samples:", Y_train.head())

# Generate predictions needed for metrics
y_pred = lr.predict(X_test)

# Regression line plot
plt.scatter(df['Hours'], df['Scores'])
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.plot(X_train, lr.predict(X_train), color='red')
plt.title('Regression Line')
plt.show()

print("Coefficient:", lr.coef_)
print("Intercept:", lr.intercept_)

# Metrics (fixed function name and defined y_pred)
mse = mean_squared_error(Y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(Y_test, y_pred)
r2 = r2_score(Y_test, y_pred)

print("MSE:", mse)
print("RMSE:", rmse)
print("MAE:", mae)
print("R2 Score:", r2)
```

## Output:
![WhatsApp Image 2026-02-04 at 4 02 22 PM (3)](https://github.com/user-attachments/assets/7fe185aa-ab74-45fe-a10b-7ff2fd011e2e)
![WhatsApp Image 2026-02-04 at 4 02 22 PM (2)](https://github.com/user-attachments/assets/d2ce8f39-46fb-48fe-a5e7-9bad69998966)
![WhatsApp Image 2026-02-04 at 4 02 22 PM (1)](https://github.com/user-attachments/assets/592c21ac-dc6b-4eff-913a-64c79e3b3250)
![WhatsApp Image 2026-02-04 at 4 02 22 PM](https://github.com/user-attachments/assets/720244ec-033d-422d-88e5-ce776d82be51)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
