import math
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sklearn
from sklearn import svm, preprocessing

# dataframe as df
df = pd.read_csv("datasets/covid_19_data_tr.csv")
df['Date'] = pd.to_datetime(df["Date"])
# df['Date'] = df['Date'].values.astype('datetime64[D]')

# to store the dates before setting them as an index
date = df["Date"]

df = df.set_index("Date")

# GETTING THE LAST MONTH'S DATA AS ANOTHER DATAFRAME
# lastmonth_df = df.tail(30).copy()

# GENERAL GRAPH
'''df.plot()  # figsize=(10,7)
plt.title('General Graph')
plt.show()'''

# LAST MONTH'S GRAPH
'''lastmonth_df.plot()  # figsize=(10,7)
plt.title('Last Month Graph')
plt.show()
'''


''' -------------------- MACHINE LEARNING (PREDICTION) -------------------- '''

X = df.drop("Day", axis=1).values
# X = preprocessing.scale(X)
y = df['Day'].values

print(y)

test_size = 30

X_train = X[:-test_size]
y_train = y[:-test_size]

X_test = X[-test_size:]
y_test = y[-test_size:]

clf = svm.SVR(kernel="linear")
clf.fit(X_train, y_train)
print(str(clf.score(X_test, y_test)))

''' -------------------- GRAPHING/VISUALIZATION -------------------- '''

fig = plt.figure(figsize=(70,25))
ax = plt.subplot2grid((1,1), (0,0))

# getting values from our dataframe(s)
confirmed = df["Confirmed"]
recovered = df["Recovered"]
death = df["Deaths"]

# filling areas between lines etc.
ax.fill_between(date, 0, confirmed, color='crimson')
ax.fill_between(date, 0, recovered, color='darkgreen')
ax.fill_between(date, 0, death, color='grey')

# rotating the X axis values for 45 degrees, Y axis values for 90 degrees
for label in ax.xaxis.get_ticklabels():
    label.set_rotation(45)

for label in ax.yaxis.get_ticklabels():
    label.set_rotation(90)

# GRAPH CREATION
plt.title('COVID-19 END DATE PREDICTION')
'''plt.xlabel('Date')
plt.ylabel('Amount')'''

# setting the interval of the x-axis (day-to-day?) time consumption???
'''start, end = ax.get_xlim()
plt.xticks(np.arange(start, end, 30))'''

plt.xticks(fontsize=32)
plt.yticks(fontsize=32)
# plt.legend()
# plt.show()
