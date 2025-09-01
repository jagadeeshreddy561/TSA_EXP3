# Ex.No: 03   COMPUTE THE AUTO FUNCTION(ACF)
### Developed By: Mallu Jagadeeswar Reddy
### Register No: 212222240059
### Date: 

### AIM:
To Compute the AutoCorrelation Function (ACF) of the infy_stock dataset and 
to determine the model
type to fit the data.
### ALGORITHM:
1. Import the necessary packages
2. Find the mean, variance and then implement normalization for the data.
3. Implement the correlation using necessary logic and obtain the results
4. Store the results in an array
5. Represent the result in graphical representation as given below.
### PROGRAM:
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
 
np.random.seed(0)
data = pd.read_csv('infy_stock1.csv')
data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values(by='Date') 
data.set_index('Date', inplace=True)
data.dropna(inplace=True)
plt.figure(figsize=(12, 6))
plt.plot( data['Volume'], label='Data')
plt.xlabel('Date')
plt.ylabel('Volume')
plt.legend()
plt.title('Volume Data')
plt.show()
data.dropna(inplace=True)
train_size = int(0.8 * len(data))
train_data = data[:train_size]
test_data = data[train_size:]
y_train = train_data['Volume']
y_test = test_data['Volume']
from statsmodels.graphics.tsaplots import plot_acf
series = data['Volume']
plot_acf(series)
plt.show()
data['Volume'].corr(data['Volume'].shift(1))
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.api import AutoReg
from sklearn.metrics import mean_absolute_error, mean_squared_error
 
lag_order = 1 
ar_model = AutoReg(y_train, lags=lag_order)
ar_results = ar_model.fit()
import statistics

y_pred = ar_results.predict(start=len(train_data), end=len(train_data) + len(test_data) - 1, dynamic=False)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
variance = np.var(y_test)

print(f'Mean Absolute Error: {mae:.2f}')
print(f'Root Mean Squared Error: {rmse:.2f}')
print(f'Variance_testing: {variance:.2f}')

plt.figure(figsize=(12, 6))
plt.plot(test_data["Volume"] ,y_test, label='Actual Volume')
plt.plot( test_data["Volume"],y_pred, label='Predicted Volume', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Volume')
plt.legend()
plt.title('Volume Prediction with Autoregressive Model')
plt.show()
```

### OUTPUT:
#### VISUAL REPRESENTATION OF DATASET:

#### AUTO CORRELATION:
![TSA_3 1](https://github.com/user-attachments/assets/2eab62c1-67fd-4ba4-898b-09f12e694471)

![TSA_3 2](https://github.com/user-attachments/assets/3eb96c0f-ee94-42e2-b4d3-19ee43705a1a)




#### VALUES OF MAE,RMSE,VARIANCE:

![TSA_3 3](https://github.com/user-attachments/assets/fee6b11d-d2af-48d6-856b-46964ef05817)


#### AUTOREGRESSIVE MODEL FOR CONSUMPTION PREDICTION
![TSA_3 4](https://github.com/user-attachments/assets/b013a8f7-a061-431b-9d04-f6ad6008d39a)


### RESULT: 
Thus, The python code for implementing auto correlation for infy_stock is successfully executed.
