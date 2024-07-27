# Fraud Detection App

## Overview

The Fraud Detection App is a Streamlit-based web application that uses a pre-trained RandomForestClassifier model to predict fraudulent transactions. The app allows users to manually input transaction data or upload a CSV or Excel file to perform predictions on multiple transactions at once.

## Features

- **Manual Entry**: Input transaction data manually to get predictions.
- **File Upload**: Upload a CSV or Excel file containing transaction data and receive predictions for each transaction.
- **Download Results**: Download the results of predictions in a CSV file.

## Setup

### Prerequisites

- Python 3.6 or higher
- Streamlit
- Pandas
- Scikit-learn
- Joblib
- NumPy

### Installation

1. Clone this repository:

    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2. Install the required packages:

    ```bash
    pip install streamlit pandas scikit-learn joblib numpy
    ```

3. Place your trained RandomForestClassifier model file in the project directory and name it `m.pkl`.

### Running the App

1. Navigate to the project directory in your terminal.

2. Run the Streamlit app:

    ```bash
    streamlit run app.py
    ```

    Replace `app.py` with the name of your Streamlit Python file if different.

3. Open a web browser and go to the URL displayed in the terminal (usually `http://localhost:8501`).

## Usage

### Manual Entry

1. On the sidebar, enter the values for the following fields:
    - Time
    - V1 to V28 (28 variables)
    - Amount

2. Click the "Predict from Manual Entry" button to see the prediction and probability for the entered data.

### File Upload

1. Click on the "Choose a file" button and select a CSV or Excel file containing the transaction data.

2. The file should contain columns named `Time`, `V1` to `V28`, and `Amount`.

3. After the file is uploaded, click the "Predict from Uploaded File" button.

4. The app will display the predictions and provide a link to download the predictions as a CSV file.

## Notes

- Ensure your file has the correct format and column names as required by the app.
- The model used for predictions should be compatible with the data structure and format expected by the app.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For any issues or questions, please open an issue in the repository or contact [your-email@example.com](mailto:your-email@example.com).
