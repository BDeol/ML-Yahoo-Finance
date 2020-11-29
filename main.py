import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression

# Read in Yahoo finance historcal stock price date for the SPY ETF
df = pd.read_csv('C:/Users/BDeol/Downloads/SPY.csv')
df.set_index('Date', inplace=True, drop=True)

# Calculate the High/Low and the Open/Close change percentage for each row
df['HL Change'] = (((df['High'] - df['Low']) / df['Low']) * 100)
df['OC Change'] = (((df['Close'] - df['Open']) / df['Open']) * 100)

df = df[['Close', 'HL Change', 'OC Change', 'Volume']]


# Set the forecast column to be the Close amount
forecast_col = 'Close'
# Replace any NA values with -99999 so the classifier disregards
df.fillna(-99999, inplace=True)
# Set how far out you want to forecast
forecast_out = 15

# Shift the Label column in the dataframe by the amount you want to forecast
df['Label'] = df[forecast_col].shift(-forecast_out)

# Drop the Lable column so you are only left with the features
X = np.array(df.drop(['Label'], 1))
# Scale the feature values to normalize the data
X = preprocessing.scale(X)
# Separate the rows you will be predicting from the ones you already have values for
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

# Remove all the na rows from the dataframe
df.dropna(inplace=True)
# Set y to the labels
y = np.array(df['Label'])

# Separate data for training and testing
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

# Create a linear regression classifier which will use all available CPU's
clf = LinearRegression(n_jobs=1)
# Fit the classifier using the training data
clf.fit(X_train, y_train)
# Get the accuracy of the test data
accuracy = clf.score(X_test, y_test)

# Forecast the close prices for the latest data
forecast_set = clf.predict(X_lately)


print(forecast_set, accuracy, forecast_out)


df['Forecast'] = np.nan

# Get the last date in the data set and convert it to datetime object
last_date = df.iloc[-1].name
last_date = datetime.datetime.strptime(last_date, '%Y-%m-%d')
print(last_date)
# Convert last date to unix format for easy incrementing
last_unix = last_date.timestamp()
print(last_unix)
# Increment the date by one date to get the first forecast date
one_day = 86400
next_unix = last_unix + one_day

# Iterate through each forecast value
for i in forecast_set:
    # Convert the forecast date to date time format
    next_date = datetime.datetime.fromtimestamp(next_unix)
    # Increment the unix date for next interation
    next_unix += one_day
    # Create the row for the forecast dates by inserting nan for all entries and then adding the forecast date back to
    # the data set in the Forecast column
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

# Plot the historical close prices
df['Close'].plot()
# Plot the forecasted close prices
df['Forecast'].plot()
# Set the labels and show the plot
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

