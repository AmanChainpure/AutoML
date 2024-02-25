import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
import seaborn as sns
import pickle
import io
from sklearn.tree import DecisionTreeClassifier
st.set_option('deprecation.showPyplotGlobalUse', False)

# Step 2: Upload data file (CSV)
def upload_data(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        st.error(f"An error occurred while uploading the data: {str(e)}")

# Display first 5 rows of the DataFrame
def display_head(data):
    try:
        st.subheader("First 5 rows of the DataFrame:")
        st.write(data.head())
    except Exception as e:
        st.error(f"An error occurred while displaying the first 5 rows: {str(e)}")

# Choose Features and Target based on user input
def choose_features(data):
    try:
        st.subheader("Select Features & Target:")
        all_columns = data.columns.tolist()

        # Display all columns with checkboxes for user selection
        selected_features = st.multiselect("Select Features", all_columns, default=all_columns[:-1])  # Exclude the last column as the default target
        target_column = st.selectbox("Select Target", all_columns)

        # Filter DataFrame based on user selection
        selected_features.append(target_column)
        filtered_data = data[selected_features]
        return filtered_data, target_column
    except Exception as e:
        st.error(f"An error occurred while choosing features and target: {str(e)}")

# Display DataFrame description
def display_description(data):
    try:
        st.subheader("DataFrame Description:")
        st.write(data.describe())
    except Exception as e:
        st.error(f"An error occurred while displaying the DataFrame description: {str(e)}")

# Display data types of each column
# Display data types of each column
def display_data_types(data):
    try:
        st.subheader("Data Types of Each Column:")
        for column in data.columns:
            dtype = data[column].dtype
            st.write(f"{column}: {dtype}")
    except Exception as e:
        st.error(f"An error occurred while displaying data types: {str(e)}")

# Convert categorical variables to numerical using Label Encoding
def label_encode_categorical(data):
    try:
        st.subheader("Label Encoding for Categorical Variables:")
        le = LabelEncoder()

        for column in data.select_dtypes(include='object').columns:
            st.write(f"Label Encoding for {column}")
            data[column] = le.fit_transform(data[column])

        return data
    except Exception as e:
        st.error(f"An error occurred during label encoding: {str(e)}")

# Impute missing values
def impute_missing_values(data):
    try:
        st.subheader("Imputing Missing Values:")
        imputer = SimpleImputer(strategy='mean')

        for column in data.columns:
            if data[column].isnull().sum() > 0:
                st.write(f"Imputing missing values for {column}")
                data[column] = imputer.fit_transform(data[column].values.reshape(-1, 1))

        return data
    except Exception as e:
        st.error(f"An error occurred during imputation of missing values: {str(e)}")

# Simple EDA (Countplot and Histogram)
def simple_eda(data):
    try:
        st.subheader("Exploratory Data Analysis:")
        for column in data.columns:
            st.subheader(f"Countplot for {column}")
            sns.countplot(x=column, data=data)
            st.pyplot()

            st.subheader(f"Histogram for {column}")
            sns.histplot(data[column], kde=True)
            st.pyplot()
    except Exception as e:
        st.error(f"An error occurred during EDA: {str(e)}")

# Heatmap (with numerical data only)
def display_heatmap(data):
    try:
        st.subheader("Heatmap:")

        # Select only numerical columns for the heatmap
        numerical_columns = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
        numerical_data = data[numerical_columns]

        correlation_matrix = numerical_data.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=.5)
        st.pyplot()
    except Exception as e:
        st.error(f"An error occurred while displaying the heatmap: {str(e)}")

# Apply Standard Scaling to features and target during training
def apply_standard_scaling_train(data, target_column):
    try:
        st.subheader("Standard Scaling:")
        scaler = StandardScaler()

        # Selecting only features for scaling
        features = data.drop(target_column, axis=1)

        for column in features.select_dtypes(include=['int64', 'float64']).columns:
            st.write(f"Standard Scaling for {column}")
            features[column] = scaler.fit_transform(features[column].values.reshape(-1, 1))

        # Combining scaled features with the target column
        scaled_data = pd.concat([features, data[target_column]], axis=1)

        return scaled_data
    except Exception as e:
        st.error(f"An error occurred during standard scaling: {str(e)}")

# Apply Standard Scaling to features only during prediction
def apply_standard_scaling_predict(data):
    try:
        st.subheader("Standard Scaling:")
        scaler = StandardScaler()

        for column in data.select_dtypes(include=['int64', 'float64']).columns:
            st.write(f"Standard Scaling for {column}")
            data[column] = scaler.fit_transform(data[column].values.reshape(-1, 1))

        return data
    except Exception as e:
        st.error(f"An error occurred during standard scaling: {str(e)}")

# Run multiple classification models and save the best model
def run_classification(X_train, X_test, y_train, y_test):
    try:
        st.subheader("AutoML Model (Classification):")
        models = {
            "Random Forest": RandomForestClassifier(),
            "Logistic Regression": LogisticRegression(),
            "Decision Tree": DecisionTreeClassifier()
        }
        best_model_name = ""
        best_accuracy = 0

        for name, model in models.items():
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            st.write(f"{name} Accuracy: {accuracy:.2f}")

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_name = name
                best_model = model

        st.write(f"Best Model: {best_model_name} (Accuracy: {best_accuracy:.2f})")

        # Display classification report of best model
        st.subheader("Classification Report:")
        best_model.fit(X_train, y_train)
        predictions = best_model.predict(X_test)
        class_report = classification_report(y_test, predictions, output_dict=True)
        st.write(pd.DataFrame(class_report).transpose())

        # Save best model
        with open(f"classification_model.pkl", "wb") as file:
            pickle.dump(best_model, file)
        st.write(f"Best model saved successfully as 'classification_model.pkl!")
    except Exception as e:
        st.error(f"An error occurred during classification: {str(e)}")


# Run multiple regression models and save the best model
def run_regression(X_train, X_test, y_train, y_test):
    try:
        st.subheader("AutoML Model (Regression):")
        models = {
            "Random Forest": RandomForestRegressor(),
            "Linear Regression": LinearRegression(),
            "Gradient Boosting": GradientBoostingRegressor()
        }
        best_model_name = ""
        best_r2 = 0

        for name, model in models.items():
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            r2 = r2_score(y_test, predictions)
            st.write(f"{name} R-squared: {r2:.2f}")

            if r2 > best_r2:
                best_r2 = r2
                best_model_name = name
                best_model = model

        st.write(f"Best Model: {best_model_name} (R-squared: {best_r2:.2f})")
        best_model = models[best_model_name]
        st.subheader(f"Best Model: {best_model_name}")
        # Save the best model as a pickle file

        with open("regression_model.pkl", "wb") as file:
            pickle.dump(best_model, file)
        st.write("Best model saved successfully as 'regression_model.pkl'!")
    except Exception as e:
        st.error(f"An error occurred during regression: {str(e)}")
# Predict using trained model for Classification
def predict_classification(data, model):
    try:
        st.subheader("Make Predictions (Classification):")
        # Apply the same preprocessing steps as done for training data
        data = label_encode_categorical(data)
        data = impute_missing_values(data)
        data = apply_standard_scaling_predict(data)

        # Display updated DataFrame after preprocessing
        st.subheader("Preprocessed Data for Prediction:")
        st.write(data)

        # Predict
        prediction = model.predict(data)
        predictions_df = pd.DataFrame(prediction, columns=["Predictions"])
        # Merge original data with predictions
        result_df = pd.concat([data.reset_index(drop=True), predictions_df], axis=1)
        # Highlight Predictions column
        result_df_styled = result_df.style.apply(lambda x: ['font-weight: bold' if x.name == "Predictions" else '' for _ in x])
        # Display the result
        st.subheader("Test Data with Predictions:")
        st.write(result_df_styled)
    except Exception as e:
        st.error(f"An error occurred during classification prediction: {str(e)}")

# Predict using trained model for Regression
def predict_regression(data, model):
    try:
        st.subheader("Make Predictions (Regression):")
        # Apply the same preprocessing steps as done for training data
        data = label_encode_categorical(data)
        data = impute_missing_values(data)
        data = apply_standard_scaling_predict(data)

        # Display updated DataFrame after preprocessing
        st.subheader("Preprocessed Data for Prediction:")
        st.write(data)

        # Predict
        prediction = model.predict(data)
        predictions_df = pd.DataFrame(prediction, columns=["Predictions"])
        # Merge original data with predictions
        result_df = pd.concat([data.reset_index(drop=True), predictions_df], axis=1)
        # Highlight Predictions column
        result_df_styled = result_df.style.apply(lambda x: ['font-weight: bold' if x.name == "Predictions" else '' for _ in x])
        # Display the result
        st.subheader("Test Data with Predictions:")
        st.write(result_df_styled)
    except Exception as e:
        st.error(f"An error occurred during regression prediction: {str(e)}")

# Main function
def main():
    try:
        st.title("AutoML Tool")

        # Left side options
        tool_selection = st.sidebar.selectbox("Select Tool", ["AutoML (Classification)", "AutoML (Regression)", "Prediction_Classification", "Prediction_Regression"])

        if tool_selection == "AutoML (Classification)":
            st.sidebar.header("AutoML Settings (Classification)")

            # Example usage
            file_path = st.sidebar.file_uploader("Upload CSV File", type=["csv"])
            if file_path:
                # Step 1: Upload data
                data = upload_data(file_path)

                # Display first 5 rows of the DataFrame
                display_head(data)

                # Allow user to choose Features and Target
                filtered_data, target_column = choose_features(data)

                # Display DataFrame description
                display_description(filtered_data)

                # Display data types of each column
                display_data_types(filtered_data)

                # Convert categorical variables to numerical using Label Encoding
                filtered_data = label_encode_categorical(filtered_data)

                # Impute missing values
                filtered_data = impute_missing_values(filtered_data)

                # Display updated DataFrame
                st.subheader("Updated DataFrame:")
                st.write(filtered_data)

                # Simple EDA (Countplot and Histogram)
                simple_eda(filtered_data)

                # Heatmap (with numerical data only)
                display_heatmap(filtered_data)

                # Apply Standard Scaling
                filtered_data = apply_standard_scaling_train(filtered_data, target_column)

                # Display updated DataFrame after Standard Scaling
                st.subheader("Updated DataFrame after Standard Scaling:")
                st.write(filtered_data)

                # Split data into training and testing sets
                X_train, X_test, y_train, y_test = train_test_split(
                    filtered_data.drop(target_column, axis=1),
                    filtered_data[target_column],
                    test_size=0.2,
                    random_state=42
                )

                # Run the model using RandomForest for Classification
                run_classification(X_train, X_test, y_train, y_test)

        elif tool_selection == "AutoML (Regression)":
            st.sidebar.header("AutoML Settings (Regression)")

            # Example usage
            file_path = st.sidebar.file_uploader("Upload CSV File", type=["csv"])
            if file_path:
                # Step 1: Upload data
                data = upload_data(file_path)

                # Display first 5 rows of the DataFrame
                display_head(data)

                # Allow user to choose Features and Target
                filtered_data, target_column = choose_features(data)

                # Display DataFrame description
                display_description(filtered_data)

                # Display data types of each column
                display_data_types(filtered_data)

                # Convert categorical variables to numerical using Label Encoding
                filtered_data = label_encode_categorical(filtered_data)

                # Impute missing values
                filtered_data = impute_missing_values(filtered_data)

                # Display updated DataFrame
                st.subheader("Updated DataFrame:")
                st.write(filtered_data)

                # Simple EDA (Countplot and Histogram)
                simple_eda(filtered_data)

                # Heatmap (with numerical data only)
                display_heatmap(filtered_data)

                # Apply Standard Scaling
                filtered_data = apply_standard_scaling_train(filtered_data, target_column)

                # Display updated DataFrame after Standard Scaling
                st.subheader("Updated DataFrame after Standard Scaling:")
                st.write(filtered_data)

                # Split data into training and testing sets
                X_train, X_test, y_train, y_test = train_test_split(
                    filtered_data.drop(target_column, axis=1),
                    filtered_data[target_column],
                    test_size=0.2,
                    random_state=42
                )

                # Run the model using RandomForest for Regression
                run_regression(X_train, X_test, y_train, y_test)

        elif tool_selection == "Prediction_Classification":
            st.sidebar.header("Prediction Settings")
            # Example usage
            model_file_path = st.sidebar.file_uploader("Upload Model File", type=["pkl"])
            if model_file_path:
                # Load the trained model
                model_file_contents = model_file_path.getvalue()
                model = pickle.load(io.BytesIO(model_file_contents))
                # Example usage
                prediction_data_file_path = st.sidebar.file_uploader("Upload Data File for Prediction", type=["csv"])
                if prediction_data_file_path:
                    # Step 1: Upload data
                    prediction_data = upload_data(prediction_data_file_path)
                    # Display first 5 rows of the DataFrame
                    display_head(prediction_data)
                    # Predict
                    predict_classification(prediction_data, model)


        elif tool_selection == "Prediction_Regression":
            st.sidebar.header("Prediction Settings")
          # Example usage
            model_file_path = st.sidebar.file_uploader("Upload Model File", type=["pkl"])
            if model_file_path:
                # Load the trained model
                model_file_contents = model_file_path.getvalue()
                model = pickle.load(io.BytesIO(model_file_contents))

                prediction_data_file_path = st.sidebar.file_uploader("Upload Data File for Prediction", type=["csv"])
                if prediction_data_file_path:
                    # Step 1: Upload data
                    prediction_data = upload_data(prediction_data_file_path)
                    # Display first 5 rows of the DataFrame
                    display_head(prediction_data)
                    # Predict
                    predict_regression(prediction_data, model)

    except Exception as e:
        st.error(f"An error occurred in the main function: {str(e)}")

if __name__ == "__main__":
    main()
