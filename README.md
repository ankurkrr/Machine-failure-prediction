
# Predictive Analysis for Manufacturing Operations

## Overview
This project implements a RESTful API for predictive analysis in manufacturing operations. The API predicts machine downtime or production defects using a machine learning model trained on manufacturing data.

## Features
- Upload a manufacturing dataset via the `/upload` endpoint.
- Train a machine learning model via the `/train` endpoint.
- Make predictions via the `/predict` endpoint.

---

## Installation

### Prerequisites
- Python 3.8 or later.
- Libraries: Flask/FastAPI, scikit-learn, pandas, numpy.

### Steps to Set Up
1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_folder>
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### Running the API
1. Start the server:
   ```bash
   uvicorn app:app --reload
   ```
   (Replace `app:app` with the filename and application instance name if different.)

2. The API will be available at:
   ```
   http://127.0.0.1:8000
   ```

---

### API Endpoints

#### **1. Upload Dataset**
- **Endpoint:** `POST /upload`
- **Description:** Upload a CSV file containing the manufacturing dataset.
- **Request Body:** Form-data with a key `file` containing the CSV file.
- **Example cURL:**
  ```bash
  curl -X POST http://127.0.0.1:8000/upload        -F "file=@sample_data.csv"
  ```
- **Response:**
  ```json
  {
    "message": "Dataset uploaded successfully."
  }
  ```

---

#### **2. Train Model**
- **Endpoint:** `POST /train`
- **Description:** Train the model on the uploaded dataset.
- **Response:**
  ```json
  {
    "accuracy": 0.92,
    "f1_score": 0.89
  }
  ```
- **Example cURL:**
  ```bash
  curl -X POST http://127.0.0.1:8000/train
  ```

---

#### **3. Make Predictions**
- **Endpoint:** `POST /predict`
- **Description:** Predict machine downtime using input features.
- **Request Body:**
  ```json
  {
    "Temperature": 80,
    "Run_Time": 120
  }
  ```
- **Response:**
  ```json
  {
    "Downtime": "Yes",
    "Confidence": 0.85
  }
  ```
- **Example cURL:**
  ```bash
  curl -X POST http://127.0.0.1:8000/predict        -H "Content-Type: application/json"        -d '{"Temperature": 80, "Run_Time": 120}'
  ```

---

## Folder Structure
```
project_folder/
│
├── app.py             # Main application file
├── model/model.pkl          # Trained model file
├── data/sample_data.csv    # Sample dataset
├── requirements.txt   # Python dependencies
└── README.md          # Documentation
```

---

## Example Dataset
Include a description or sample of the dataset you're using:
| Machine_ID | Temperature | Run_Time | Downtime_Flag |
|------------|-------------|----------|---------------|
| 1          | 75          | 100      | 0             |
| 2          | 85          | 120      | 1             |

---

## Testing
- Use Postman or cURL to test the endpoints locally.
- Ensure the model is trained before making predictions.

---

## Contributing
Feel free to fork this repository and submit pull requests for improvements.

---

## License
This project is licensed under the MIT License.
