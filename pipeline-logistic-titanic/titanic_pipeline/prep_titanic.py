# Import libraries
import os
import argparse
import pandas as pd
import joblib
from azureml.core import Run
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from azureml.core import Workspace, Dataset, Datastore
from azureml.data.datapath import DataPath
from azureml.core.authentication import ServicePrincipalAuthentication

svc_pr = ServicePrincipalAuthentication(
    tenant_id="aaf80b90-a7b0-4d67-a45e-9f667ca00d2a",
    service_principal_id="f42dabcf-89aa-49b4-934e-151f64127f09",
    service_principal_password="rJ68Q~3LC1_dBnuNYmJ0UDCu7tWFH0QXnEBHicA0") ## BAD Practice, but testing right now. use environment varibale instead or azure secret vault


# Get parameters
parser = argparse.ArgumentParser()
parser.add_argument("--input-data", type=str, dest='raw_dataset_id', help='raw dataset')
parser.add_argument('--prepped-data', type=str, dest='prepped_data', default='prepped_data', help='Folder for results')
args = parser.parse_args()
save_folder = args.prepped_data

# Need a better way to define workspace as opposed to hard code
subscription_id = '0fc728de-7dd8-42de-97fb-5ff957b4f4f1'
resource_group = 'azmachinelearning'
workspace_name = 'azmachinelearning'


# Authenticate via service principal
ws= Workspace(subscription_id, resource_group, workspace_name, auth = svc_pr)

default_ds = ws.get_default_datastore()


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
# Scaling isn't necessary for decision trees
scaler = MinMaxScaler()
num_cols = ['Age','Fare']
df[num_cols] = scaler.fit_transform(df[num_cols])

df['Sex'] = df['Sex'].replace({'male':1,'female':0})

# Log processed rows
row_count = (len(df))
run.log('processed_rows', row_count)

# Save the prepped data
print("Saving Data...")
os.makedirs(save_folder, exist_ok=True)
save_path = os.path.join(save_folder,'data.csv')
df.to_csv(save_path, index=False, header=True)

# Saving Scalar File
os.makedirs('outputs', exist_ok=True)
scaler_file = os.path.join('outputs', 'titanic_scaler.pkl')
joblib.dump(value=scaler, filename=scaler_file)


default_ds = ws.get_default_datastore()
folder_data = Dataset.File.upload_directory(src_dir='outputs',
                              target=DataPath(default_ds, 'titanic-data/')
                              )

try:
    folder_data.register(workspace=ws, name='titanic_scaler.pkl')
except:
    print('file already there')

# End the run
run.complete()
