# Import libraries
import os
import argparse
import pandas as pd
from azureml.core import Run
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder

# Get parameters
parser = argparse.ArgumentParser()
parser.add_argument("--input-data", type=str, dest='raw_dataset_id', help='raw dataset')
parser.add_argument('--prepped-data', type=str, dest='prepped_data', default='prepped_data', help='Folder for results')
args = parser.parse_args()
save_folder = args.prepped_data

# Get the experiment run context
run = Run.get_context()

# load the data (passed as an input dataset)
print("Loading Data...")
df = run.input_datasets['raw_data'].to_pandas_dataframe()

# Log raw row count
row_count = (len(df))
run.log('raw_rows', row_count)

# remove nulls
df = df.dropna()

# Normalize the numeric columns
scaler = MinMaxScaler()
num_cols = ['Age','Fare']
#df[num_cols] = scaler.fit_transform(df[num_cols])

df['Sex'] = df['Sex'].replace({'male':1,'female':0})

# Log processed rows
row_count = (len(df))
run.log('processed_rows', row_count)

# Save the prepped data
print("Saving Data...")
os.makedirs(save_folder, exist_ok=True)
save_path = os.path.join(save_folder,'data.csv')
df.to_csv(save_path, index=False, header=True)

# End the run
run.complete()
