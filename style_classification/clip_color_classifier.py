import pandas as pd
import pickle

data_path = r'data/2001_A_SPACE_ODYSSEY.csv'
df = pd.read_csv(data_path)
color_data = df.drop(columns=['fullpath', 'length'])
output_df = df[['fullpath', 'length']]

clf = pickle.load(open('TrainedRandomForestClassifier.p', 'rb'))
print('Done Loading')

y_pred = clf.predict(color_data)
pred_colors_df = pd.concat([pd.DataFrame([y_pred[i]], columns=['color']) for i in range(len(y_pred))],
                           ignore_index=True)

output_df = output_df.join(pred_colors_df)
output_df.to_csv(f'data/2001_A_SPACE_ODYSSEY-labeled.csv', index=False)
