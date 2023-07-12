# azureml-public
Running machine learning workloads on the AzureML platform. 

### Designer
- [Titanic Classification](designer-pipeline-decisiontrees-titanic) (azureml): [Original publication](https://medium.com/@manwill/getting-started-with-azureml-pipeline-and-decision-trees-9f1d19e2b139). Using AzureML Designer, utilize the canvas and interface to lay and connect components. ['set-up-titanic.ipynb](designer-pipeline-decisiontrees-titanic/set-up-titanic.ipynb) is used to fetch titanic data and sink to the AzureML datastore and also provision a compute cluster to run the pipeline. 

### Script Mode
- [Hyperdrive](link) (azureml): Use AzureML Hyperdrive to perform hyperparameter tuning on 'learning rate' and 'number of estimators' for sklearn GradientBoostingClassifier. The classifier is used to predict titanic passenger survivability. Run workload using AzureML and monitor via training job.
- [Decision Trees on AzureML - Titanic](pipeline-decisiontrees-titanic) (azureml): [Original publication](https://medium.com/@manwill/getting-started-with-azureml-pipeline-and-decision-trees-9f1d19e2b139). Create an end-to-end pipeline that fetch titanic data, prep data, and train a Decisions Trees model. Monitor via training job and deploy model for inferencing via API endpoint.
- [Logistic Regression on AzureML - Titanic](pipeline-logistic-titanic) (azureml): Create an end-to-end pipeline that fetch titanic data, prep data, and train a Logistic model. Monitor via training job and deploy model for inferencing via API endpoint.
- [Linear Regression on AzureML - Medical Charges](regression_insurance) (azureml): [Original publication](https://medium.com/@manwill/predicting-medical-charges-using-azureml-and-sklearn-linear-regression-9231d8e23b3c)Create an end-to-end pipeline that fetch medical charge data, prep data, and train a linear regression model. Monitor via training job and deploy model for inferencing via API endpoint. 
