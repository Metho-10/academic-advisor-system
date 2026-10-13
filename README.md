
# AI-Based Academic Advisor System

## Project Description

This project combines standard software engineering practices
and machine learning to build an academic advising web application.

Students enter academic information and receive a predicted
support category with practical study recommendations.

## Technologies

- Python
- Git and GitHub
- Visual Studio Code
- Google Colab
- Pandas and scikit-learn
- Pickle-compatible model serialization using joblib
- Flask
- HTML and CSS
- Render

## Features

- Academic information input
- Machine learning classification
- Input validation
- Academic support recommendations
- Responsive web interface

## Model Training

The Random Forest model is trained in Google Colab using
synthetic demonstration data.

The resulting model is saved as academic_advisor_model.pkl.

## Installation

1. Clone the repository.
2. Create and activate a Python virtual environment.
3. Install dependencies using pip install -r requirements.txt.
4. Ensure academic_advisor_model.pkl is in the project root.
5. Run python app.py.
6. Open http://127.0.0.1:5000.

## Limitations

The model uses synthetic data and has not been validated
for real-world academic decisions. Predictions should be
treated as demonstrations, not official university assessments.
