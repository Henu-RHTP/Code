import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

data_url = "all_data-2024-6-13.csv"

raw_df = pd.read_csv(data_url, sep=",",skiprows=[0], header=None)
X=raw_df[raw_df.columns[3:]]
# X=X.drop([10,],axis=1)
# X=X[[3,4,5,11,13,15,16,17,20,22,23,25,28,37,39,40,44]]
# X=X[[5]]
# X = sm.add_constant(X)
y=raw_df[raw_df.columns[2]]


# X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
X_train=X[:123]
X_test=X[123:]
y_train=y[:123]
y_test=y[123:]
X_train=sm.add_constant(X_train)
X=sm.add_constant(X)
model=sm.OLS(y,X)

# model=sm.WLS(y_train,X_train)
result=model.fit()
print(result.summary())
# x_1=X_train.dot(result.params)
# x_2=X_test.dot(result.params)
x_3=X.dot(result.params)

# y_pre=result.predict(X_test)
y_pre=result.fittedvalues

# print(y_test)
z_da=pd.DataFrame([y,y_pre])
z_da.to_excel("z_da_1.xlsx")
print(y_pre)
print(z_da)
# r2=r2_score(y_test,y_pre)
r2=r2_score(y,y_pre)
print("r2:",r2)

f, ax = plt.subplots()

# plt.scatter(x_2,y_test)
# plt.plot(x_2,y_pre)
plt.scatter(x_3,y)
plt.plot(x_3,y_pre)
plt.text(.01, .99, 'r2 : {:0.2f}'.format(r2),
         ha='left', va='top', transform=ax.transAxes)
plt.show()