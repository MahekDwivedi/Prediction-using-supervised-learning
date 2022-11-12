 # Prediction using Supervised ML (task-1)
  
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("http://bit.ly/w-data")
df

sns.scatterplot(data =df , x="Scores" , y="Hours")
plt.xlabel("Percentage Scores")
plt.ylabel("No. of Study Hours")
plt.title("Hours vs Percentage")

# SPLITTING TRAINING AND TESTING DATASET
x = df.iloc[:,:-1].values
y = df.iloc[:,1].values

print(x.shape)
y.shape

from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test=  train_test_split( x , y ,test_size=0.3 , random_state=0)

# TRAINING MODEL

from sklearn.linear_model import LinearRegression
mymodel= LinearRegression()
mymodel.fit(x_train , y_train)
print(mymodel)
predictions= mymodel.predict(x_test)

X= df['Scores']
np.polyfit(X ,y , deg=1)

values =  np.linspace(0 ,100, 25)
line = 0.0525585*values + 4.573573
print( line )

# PLOTTING PREDICTED VALUES

sns.scatterplot(data =df , x="Scores" , y="Hours")

plt.xlabel("Percentage Scores")
plt.ylabel("No. of Study Hours")
plt.title("Hours vs Percentage")
# Plotting the regression line
plt.plot( X , line , color="red")

# COMPARING PREDICTED AND ACTUAL VALUES

data = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})  
print(data)

#Predicting output for value 9.48
predicted_score = mymodel.predict([[9.48]])
print("No of Hours : 9.48")
print("predicted score : ",predicted_score[0])

# FIND ABSOLUTE ERROR 

from sklearn.metrics import mean_absolute_error , mean_squared_error
df['Scores'].mean()

mean_absolute_error( y_test , predictions )

np.sqrt (mean_squared_error( y_test , predictions))

test_residual = y_test - predictions
test_residual 
