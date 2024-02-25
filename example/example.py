import sys
sys.path.append("../asbagging")
import pandas as pd
from sklearn.model_selection import train_test_split
from model.asbagging_FSS import asbagging_FSS
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
import json
from model.config import Config


my_data = pd.read_csv("example/DARWIN.csv")
my_label = my_data["class"]

my_data=my_data.drop('ID',axis=1)
my_data=my_data.drop('class',axis=1)
le=LabelEncoder()
my_label=le.fit_transform(my_label)

def remove_all_zero_col(df):
    """Remove columns that contain all zeros"""
    df = df.copy()
    for col in df.columns:
        if (df[col] == 0).all():
            df.drop(col, axis=1, inplace=True)
    return df
my_data=remove_all_zero_col(my_data)
ms = MinMaxScaler()
my_data = ms.fit_transform(my_data)

train_data, valid_data, train_label, valid_label = train_test_split(my_data, my_label, test_size=0.2, random_state=42)

with open("config.json", "r", encoding="utf-8") as f:
    config = Config(**json.load(f))

model=asbagging_FSS(config,bootstrap=True)
model.fit(train_data, train_label)

#Predict labels for training data
train_result=model.predict(train_data)
train_count=0
for i in range(train_result.size):
    if train_result[i]==train_label[i]:
        train_count=train_count+1

#Predict labels for validation data
valid_result=model.predict(valid_data)
valid_count=0
for i in range(valid_result.size):
    if valid_result[i]==valid_label[i]:
        valid_count=valid_count+1

print('train_accuracy',train_count/train_result.size)
print('valid_accuracy',valid_count/valid_result.size)