import json
import joblib
import numpy as np
import os
#import azureml.core
#from azureml.core.authentication import ServicePrincipalAuthentication
#from azureml.core import Workspace, Dataset, Datastore
#from azureml.data.datapath import DataPath


import os
# Called when the service is loaded
def init():
    global model
    # Get the path to the deployed model file and load it
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'titanic_model.pkl')
    model = joblib.load(model_path)
    """
    svc_pr = ServicePrincipalAuthentication(
        tenant_id="aaf80b90-a7b0-4d67-a45e-9f667ca00d2a",
        service_principal_id="f42dabcf-89aa-49b4-934e-151f64127f09",
        service_principal_password="rJ68Q~3LC1_dBnuNYmJ0UDCu7tWFH0QXnEBHicA0") ## BAD Practice, but testing right now. use environment varibale instead or azure secret vault
    
    # Need a better way to define workspace as opposed to hard code
    subscription_id = '0fc728de-7dd8-42de-97fb-5ff957b4f4f1'
    resource_group = 'azmachinelearning'
    workspace_name = 'azmachinelearning'
    


    # Authenticate via service principal
    ws= Workspace(subscription_id, resource_group, workspace_name, auth = svc_pr)
    
    # Download scalar from datastore
    #path = 'titanic-data/titanic_scaler.pkl'
    #datastore = Datastore.get(ws, "workspaceblobstore")
    #scalar_file = Dataset.File.from_files(path=(datastore, path))

    #scalar_file.download(target_path='./titanic_service')
    #scalar = joblib.load('titanic_service/titanic_scaler.pkl')
    """

# Called when a request is received
def run(raw_data):
    # Get the input data as a numpy array
    data = np.array(json.loads(raw_data)['data'])
    
    ## Perform Scalar Transformation
    """
    x_transformed = scalar.transform(np.array(data)[:,[0,2]]).tolist() # transforming age and fare    
    for n in range(data.shape[0]):
        data[n][0] = x_transformed[n][0] # Age 
        data[n][2] = x_transformed[n][1] # Fare
    """

    # Get a prediction from the model
    predictions = model.predict(data)
    # Get the corresponding classname for each prediction (0 or 1)
    classnames = ['Non-Survived', 'Survived']
    predicted_classes = []
    for prediction in predictions:
        predicted_classes.append(classnames[prediction])
    # Return the predictions as JSON
    return json.dumps(predicted_classes)
