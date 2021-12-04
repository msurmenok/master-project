import numpy
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from joblib import dump, load

filepath1 = 'dataset_small_0.csv'
filepath2 = 'dataset_small_1.csv'
df1 = pd.read_csv(filepath1)
df2 = pd.read_csv(filepath2)

df = pd.concat([df1, df2])
df = df[df['algorithm'] == 'MemeticExperimental8']

# print(df.columns)

X = df[['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7']].to_numpy()
y = df[['average_total_time']].to_numpy()

# split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y)
y_train = y_train.ravel()
y_test = y_test.ravel()
# print(X_train)
# print(X_test)

# normalize dataset
# create model and train
# test prediction
# scaler = preprocessing.StandardScaler().fit(X_train)
pipe = make_pipeline(StandardScaler(),
                     MLPRegressor(max_iter=1000, verbose=True))
pipe.fit(X_train, y_train)

print(pipe.score(X_train, y_train))
print(pipe.score(X_test, y_test))

predicted = pipe.predict(X_test)
print(predicted)

for i in range(len(y_test)):
    if i == 200:
        break
    print(f'test {y_test[i]}, predicted {predicted[i]}')

# save model to the file
dump(pipe, 'mlmodel.joblib')

# load
# pipe = load('mlmodel.joblib')
#
# objectives = np.array((50.0, 17.0, 55151.0, 19842.330306, 17.0, 26.289513, 38790.0)).reshape(1, -1)
# predicted_value = pipe.predict(objectives)
# print(predicted_value)
# print(pipe.predict(np.array((50.0, 17.0, 50924.0, 19082.874343, 18.0, 26.289513, 39065.0)).reshape(1, -1)))
#
# print(pipe.predict(np.array((49.0, 17.0, 56652.0, 44071.48237, 19.0, 25.857582999999998, 37774.0)).reshape(1, -1)))
