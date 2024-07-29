import streamlit as st
import pandas as pd
import pickle  # Use pickle instead of joblib
import numpy as np
import base64
import os

# Function to load the model using pickle
def load_model(model_path):
    try:
        with open(model_path, 'rb') as model_file:
            model = pickle.load(model_file)
        return model
    except FileNotFoundError:
        st.error("Model file not found. Please check the file path.")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        return None

# Load your model
model_path = os.path.join(os.getcwd(), 'm.pkl')  # Adjust path as needed
model = load_model(model_path)

# Function to make predictions on a single observation
def predict_fraud(time, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10,
                  v11, v12, v13, v14, v15, v16, v17, v18, v19, v20,
                  v21, v22, v23, v24, v25, v26, v27, v28, amount):
    if model is None:
        st.error("Model is not loaded. Please check the model file.")
        return None, None

    input_data = np.array([[time, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10,
                            v11, v12, v13, v14, v15, v16, v17, v18, v19, v20,
                            v21, v22, v23, v24, v25, v26, v27, v28, amount]])
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[:, 1]  # For binary classification
    return prediction[0], probability[0]

# Main function to run the Streamlit app
def main():
    st.title('Fraud Detection App')

    # Sidebar for manual entry
    st.sidebar.subheader('Manual Entry')
    time = st.sidebar.number_input('Time', value=0.0)
    v1 = st.sidebar.number_input('V1', value=0.0)
    v2 = st.sidebar.number_input('V2', value=0.0)
    v3 = st.sidebar.number_input('V3', value=0.0)
    v4 = st.sidebar.number_input('V4', value=0.0)
    v5 = st.sidebar.number_input('V5', value=0.0)
    v6 = st.sidebar.number_input('V6', value=0.0)
    v7 = st.sidebar.number_input('V7', value=0.0)
    v8 = st.sidebar.number_input('V8', value=0.0)
    v9 = st.sidebar.number_input('V9', value=0.0)
    v10 = st.sidebar.number_input('V10', value=0.0)
    v11 = st.sidebar.number_input('V11', value=0.0)
    v12 = st.sidebar.number_input('V12', value=0.0)
    v13 = st.sidebar.number_input('V13', value=0.0)
    v14 = st.sidebar.number_input('V14', value=0.0)
    v15 = st.sidebar.number_input('V15', value=0.0)
    v16 = st.sidebar.number_input('V16', value=0.0)
    v17 = st.sidebar.number_input('V17', value=0.0)
    v18 = st.sidebar.number_input('V18', value=0.0)
    v19 = st.sidebar.number_input('V19', value=0.0)
    v20 = st.sidebar.number_input('V20', value=0.0)
    v21 = st.sidebar.number_input('V21', value=0.0)
    v22 = st.sidebar.number_input('V22', value=0.0)
    v23 = st.sidebar.number_input('V23', value=0.0)
    v24 = st.sidebar.number_input('V24', value=0.0)
    v25 = st.sidebar.number_input('V25', value=0.0)
    v26 = st.sidebar.number_input('V26', value=0.0)
    v27 = st.sidebar.number_input('V27', value=0.0)
    v28 = st.sidebar.number_input('V28', value=0.0)
    amount = st.sidebar.number_input('Amount', value=0.0)

    # Button to make prediction for manual entry
    if st.sidebar.button('Predict from Manual Entry'):
        prediction, probability = predict_fraud(time, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10,
                                                v11, v12, v13, v14, v15, v16, v17, v18, v19, v20,
                                                v21, v22, v23, v24, v25, v26, v27, v28, amount)
        st.sidebar.write('Prediction:', prediction)
        st.sidebar.write('Probability:', probability)

    # Main section for file upload
    st.subheader('Upload CSV or Excel File')
    uploaded_file = st.file_uploader('Choose a file', type=['csv', 'xlsx'])

    if uploaded_file is not None:
        # Display uploaded file
        st.write('Uploaded file:')
        try:
            # Specify encoding as 'latin-1' or 'ISO-8859-1' depending on your file's encoding
            df = pd.read_csv(uploaded_file, encoding='latin-1')  # Load CSV data into DataFrame
            st.dataframe(df)

            # Button to make prediction for uploaded file
            if st.button('Predict from Uploaded File'):
                predictions = []
                probabilities = []

                for index, row in df.iterrows():
                    time = row['Time']
                    v1 = row['V1']
                    v2 = row['V2']
                    v3 = row['V3']
                    v4 = row['V4']
                    v5 = row['V5']
                    v6 = row['V6']
                    v7 = row['V7']
                    v8 = row['V8']
                    v9 = row['V9']
                    v10 = row['V10']
                    v11 = row['V11']
                    v12 = row['V12']
                    v13 = row['V13']
                    v14 = row['V14']
                    v15 = row['V15']
                    v16 = row['V16']
                    v17 = row['V17']
                    v18 = row['V18']
                    v19 = row['V19']
                    v20 = row['V20']
                    v21 = row['V21']
                    v22 = row['V22']
                    v23 = row['V23']
                    v24 = row['V24']
                    v25 = row['V25']
                    v26 = row['V26']
                    v27 = row['V27']
                    v28 = row['V28']
                    amount = row['Amount']

                    prediction, probability = predict_fraud(time, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10,
                                                            v11, v12, v13, v14, v15, v16, v17, v18, v19, v20,
                                                            v21, v22, v23, v24, v25, v26, v27, v28, amount)

                    predictions.append(prediction)
                    probabilities.append(probability)

                df['Prediction'] = predictions
                df['Probability'] = probabilities
                st.write('Predictions:')
                st.dataframe(df)

                # Download link for predictions
                csv = df.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()  # B64 encoding
                href = f'<a href="data:file/csv;base64,{b64}" download="predictions.csv">Download Predictions CSV File</a>'
                st.markdown(href, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error: {e}")

# Run the main function to start the app
if __name__ == '__main__':
    main()
