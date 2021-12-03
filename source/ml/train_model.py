from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

filepath = 'dataset_small_0.csv'
df = pd.read_csv(filepath)
print(df.columns)
X = df[['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7']].to_numpy()
y = df[['average_total_time']].to_numpy()

# split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y)
y_train = y_train.ravel()
y_test = y_test.ravel()
print(X_train)
print(X_test)

# normalize dataset
# create model and train
# test prediction
# scaler = preprocessing.StandardScaler().fit(X_train)
pipe = make_pipeline(StandardScaler(),
                     MLPRegressor(max_iter=1000, verbose=True))
pipe.fit(X_train, y_train)
print(pipe.score(X_test, y_test))
predicted = pipe.predict(X_test)
print(predicted)

for i in range(len(y_test)):
    if i == 200:
        break
    print(f'test {y_test[i]}, predicted {predicted[i]}')
