# Dialogflow Stratified K-fold Cross-Validation

This project is designed to complete a Python Stratified K-fold Cross-validation process on Google Dialogflow Chabot free and ES versions. Then analyze the cross-validation results to improve the chatbots natural language understanding (NLU) performance.

The overall process to analyze the Dialogflow NLU is:
1. **Get** k-fold cross-validation results data
2. **Build** reports from the k-fold cross-validation results
3. **Analyze** the results in web app, Excel or other tool.

## Requirements
The project needs two Dialogflow agents. The agent that contains all the training phrases and entities will be refered to as the Main Agent. The agent 
that receivies copies of the training phrases and entities from the Main Agent will be refered to as the Test Agent. Below is a list of requirements that will need to be fullfiled before running the code below.

1. **Two Google Cloud Projects**
    + For the Main Agent and the Test Agent 
2. **Two Google Cloud Service Accounts for Each Project**
    + The Main Agent service account needs access to the Dialogflow API
    + The Test Agent service account needs permission to access the Main Agent Service Account. This enables you to only need one service account.
3. **A Google Cloud JSON credentials file** of the Main Agent's service account downloaded to your local machine and placed in the main_folder. This file should be named "google_cloud_credentials.json" to avoid having to change the name in the code.  
4. **The correct Python libraries installed** on your computer in a python envivronment. See requirements.txt or environment.yaml
5. **The correct values intialized** in 1_process_cross_validate and 2_build_cross_validate_reports


## Files
#### Jupyter Files:
These files are good for experimenting and testing.
1. **jupyter_1_process_cross_validate.ipynb**: A Python Jupyter notebook that completes k-fold cross-validations using Dialogflows API to obtain data for analysis
2. **jupyter_2_build_cross_validate_reports.ipynb**: A Python Jupyter notebook that to build tables from k-fold cross-validation data to analyze
3. **jupyter_3_app_analyze_intents.ipynb**: A Python Jupyter notebook to test and add new features to the web app that analyzes intents in a cell-by-cell method. 
4. **jupyter_4_test_dialogflow_api.ipynb**: A Python Jupyter notebook to test invidual functions of the Dialogflow API before incorporating them in a process

#### Python Files
1. **app_analyze_intents.py**: A streamlit web app to analyze the k-fold cross-validation report

#### Other
1. **build_cross_validate_reports.log**: A log file for 2_build_cross_validate_reports
2. **data**: The folder where input data, processed data and output data is stored. 
3. **process_cross_validated.log**: A log file for 1_process_cross_validate
4. **requirements.txt**: The file the contains the python libraries requirements to run the program.