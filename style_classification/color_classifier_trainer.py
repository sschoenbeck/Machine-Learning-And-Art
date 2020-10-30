import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import pickle
import time

start_time = time.time()
df = pd.read_csv(r'data/created_colored_images_5.csv')
y = df['color']
X = df.drop(columns=['color', 'fullpath', 'length'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
y_test_list = y_test.to_list()
clf = RandomForestClassifier(n_estimators=100, n_jobs=-1)

loading_time = time.time()
print(f'Loading Time: {loading_time - start_time}')

print('Training')
clf.fit(X_train, y_train)
training_time = time.time()

print(f'Training Time: {training_time - loading_time}')
print('Predicting')
y_pred = clf.predict(X_test)

print(f'Accuracy : {accuracy_score(y_test, y_pred)}')
predicting_time = time.time()
print(f'Training Time: {predicting_time - training_time}')

pickle.dump(clf, open('TrainedRandomForestClassifier.p', 'wb'))
