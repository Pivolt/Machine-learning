import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the JSON data
with open('D:\\MachineLearning\\baza_pozycji.json') as f:
    data = json.load(f)

# Convert JSON data to pandas DataFrame
df = pd.json_normalize(data, record_path=['skan'], meta=['XY'])
df = pd.concat([df.drop(columns=['XY']), pd.json_normalize(df['XY']).astype(float)], axis=1)

# rows = []
# for entry in data:
#     xy = entry['XY']
#     skan = entry['skan']
#     row = {
#         'X': xy['X'],
#         'Y': xy['Y'],
#         'Z': xy['Z']
#     }
#     for i, access_point in enumerate(skan):
#         row[f'MAC_{i+1}'] = access_point['MAC']
#         row[f'RSSI_{i+1}'] = access_point['RSSI']
#     rows.append(row)

# df = pd.DataFrame(rows)


# Encode the MAC column using one-hot encoding
#enc = OneHotEncoder()
#MAC_encoded = enc.fit_transform(df[['MAC']])
#MAC_encoded = pd.DataFrame(MAC_encoded.toarray(), columns=enc.get_feature_names_out(['MAC']))
#df = pd.concat([df.drop('MAC', axis=1), MAC_encoded], axis=1)

# Encode the MAC column using label encoding
enc = LabelEncoder()
df['MAC_encoded'] = enc.fit_transform(df['MAC'])
df.drop(columns=['MAC'], inplace=True)

# Split the data into training and testing sets
X = df[['MAC_encoded', 'RSSI']]
y = df[['X', 'Y']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create the neural network model
#model = MLPRegressor(hidden_layer_sizes=(20, 10), max_iter=10000, activation='relu', solver='adam', alpha=0.0001, 
#                     batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, 
#                     momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, 
#                     beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10, max_fun=15000, 
#                     verbose=False, warm_start=False)
#model.out_activation_ = 'identity'


model = RandomForestRegressor(n_estimators=100, random_state=42)

model.fit(X_train, y_train)

# Make predictions on test data and evaluate model performance
y_pred = model.predict(X_test)
score = model.score(X_test, y_test)
print('Model Score:', score)
#print(y_pred)