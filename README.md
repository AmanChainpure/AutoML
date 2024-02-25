# AutoML
AutoML is a Streamlit based tool used to automate conventional Machine Learning Classification & Regression problems

This tool can be used for any classification or regression data.
Steps:

- Select tool: AutoML Classification (or) AutoML Regression
- Upload csv file
- then, we can see first five rows of df
- then there is option window where we can select features and target
- df description
- simple eda
- Data Types of Each Column, also tell if they are not uniform
- then do Label Encoding for Categorical Variables
- Imputing Missing Values for numerical using mean and categorical using mode
- show first five rows of updated df
- Heatmap for numerical values of updated df
- Scaling for updated df
- Updated DataFrame after Standard Scaling first 5 rows
- then create multiple models for Classification or Regression
- find out which model has best accuracy
- give Classification or Regression report
- Click on tool: Classification Prediction (or) Regression prediction
- save the model as a pickle file
- Now, upload the saved pkl file
- Upload test data, get predictions
