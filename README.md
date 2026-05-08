# Loan Default Prediction API

A machine learning API built with FastAPI that predicts the likelihood of loan default based on applicant data. The model uses XGBoost classifier trained on historical loan data to provide probability scores for loan approval decisions.

## Features

- **FastAPI Backend**: High-performance REST API for real-time predictions
- **XGBoost Model**: Advanced gradient boosting algorithm for accurate predictions
- **Input Validation**: Comprehensive validation of input parameters with meaningful error messages
- **Feature Engineering**: Automatic calculation of loan-to-income ratio
- **Probability Outputs**: Returns both prediction class and confidence probability
- **Heroku Ready**: Configured for cloud deployment

## Dataset

The model is trained on the `Loan_Default.csv` dataset containing:
- Age (numeric or range format)
- Income
- Loan amount
- Credit score
- Default status (target variable)

## Model Performance

The XGBoost model achieves:
- High accuracy on test data
- Strong ROC-AUC score for probability predictions
- Balanced handling of default vs non-default cases

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd loan-default-prediction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure the model file `loan_model.pkl` is present in the root directory.

## Usage

### Local Development

Run the API server:
```bash
uvicorn app:app --reload
```

The API will be available at `http://localhost:8000`

### API Documentation

Visit `http://localhost:8000/docs` for interactive Swagger UI documentation.

## API Endpoints

### POST /predict

Predicts loan default probability based on applicant data.

**Request Body:**
```json
{
  "age": 35,
  "income": 50000,
  "loan_amount": 200000,
  "credit_score": 650
}
```

**Response:**
```json
{
  "prediction": "No Default",
  "probability": 0.78
}
```

**Validation Rules:**
- Age: 18-100
- Income: > 0
- Loan amount: > 0 (max 10,000,000)
- Credit score: 300-850
- Income sanity check: minimum 1000

## Model Training

To retrain the model with new data:

1. Update the `Loan_Default.csv` file
2. Run the training script:
```bash
python model_prediction.py
```
3. The new model will be saved as `loan_default_model.pkl` (rename to `loan_model.pkl` for API compatibility)

## Deployment

### Heroku

The project includes a `Procfile` for Heroku deployment:

```bash
heroku create your-app-name
git push heroku main
```

### Docker

Create a Dockerfile for containerized deployment:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Project Structure

```
├── app.py                 # FastAPI application
├── model_prediction.py    # Model training script
├── loan_model.pkl         # Trained XGBoost model
├── Loan_Default.csv       # Training dataset
├── requirements.txt       # Python dependencies
├── Procfile              # Heroku deployment config
└── README.md             # This file
```

## Dependencies

- fastapi: Web framework
- uvicorn: ASGI server
- pandas: Data manipulation
- scikit-learn: Machine learning utilities
- xgboost: Gradient boosting classifier
- joblib: Model serialization
- pydantic: Data validation

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.