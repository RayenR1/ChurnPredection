import logging
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),  # Enregistre les logs dans un fichier
        logging.StreamHandler()  # Affiche les logs dans la console
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI application
app = FastAPI()

# Configurer les origines autorisées
origins = [
    "http://localhost:5500",
    "http://127.0.0.1:5500",
    "http://localhost:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (CSS, JS, images)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve the HTML file (api2.html)
@app.get("/", response_class=HTMLResponse)
async def read_html():
    try:
        with open("static/api2.html", "r") as f:
            return HTMLResponse(content=f.read())
    except Exception as e:
        logger.error("Error loading HTML file: %s", e)
        raise HTTPException(status_code=500, detail="An error occurred while loading the HTML file.")

# Load reference data
try:
    state_reference = pd.read_csv('state_reference.csv')
    international_plan_reference = pd.read_csv('international_plan_reference.csv')
    voice_mail_plan_reference = pd.read_csv('voice_mail_plan_reference.csv')
    logger.info("Reference data loaded successfully.")
except Exception as e:
    logger.error("Error loading reference data: %s", e)
    raise

# Load pre-trained models and scaler
try:
    scaler_loaded = joblib.load('minmax_scaler.pkl')
    model_decision_tree = joblib.load('DecisionTree.pkl')
    model_random_forest = joblib.load('RandomForest.pkl')
    logger.info("Models and scaler loaded successfully.")
except Exception as e:
    logger.error("Error loading models or scaler: %s", e)
    raise

# Define the input data model
class InputData(BaseModel):
    State: str
    Account_length: int
    Area_code: int
    International_plan: str
    Voice_mail_plan: str
    Number_vmail_messages: int
    Total_day_calls: int
    Total_day_charge: float
    Total_eve_calls: int
    Total_eve_charge: float
    Total_night_calls: int
    Total_night_charge: float
    Total_intl_calls: int
    Total_intl_charge: float
    Customer_service_calls: int

# Endpoint for making a prediction
@app.post("/predict")
async def predict(data: InputData):
    logger.info("Received input data: %s", data.dict())
    try:
        # Convert input data to a dictionary
        input_dict = data.dict()

        # Rename columns to match the training data
        input_dict = {
            'State': input_dict.pop('State'),
            'Account length': input_dict.pop('Account_length'),
            'Area code': input_dict.pop('Area_code'),
            'International plan': input_dict.pop('International_plan'),
            'Voice mail plan': input_dict.pop('Voice_mail_plan'),
            'Number vmail messages': input_dict.pop('Number_vmail_messages'),
            'Total day calls': input_dict.pop('Total_day_calls'),
            'Total day charge': input_dict.pop('Total_day_charge'),
            'Total eve calls': input_dict.pop('Total_eve_calls'),
            'Total eve charge': input_dict.pop('Total_eve_charge'),
            'Total night calls': input_dict.pop('Total_night_calls'),
            'Total night charge': input_dict.pop('Total_night_charge'),
            'Total intl calls': input_dict.pop('Total_intl_calls'),
            'Total intl charge': input_dict.pop('Total_intl_charge'),
            'Customer service calls': input_dict.pop('Customer_service_calls'),
        }

        # Convert input data to a DataFrame
        input_data = pd.DataFrame([input_dict])
        input_data.columns = input_data.columns.str.replace('_', ' ')
        logger.info("Input data converted to DataFrame: %s", input_data)

        # Define numerical and categorical columns
        numerical_cols = input_data.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = input_data.select_dtypes(include=['object']).columns.tolist()

        # Load mappings for categorical variables
        state_mapping = dict(zip(state_reference['State'], state_reference['State_Code']))
        international_plan_mapping = dict(zip(international_plan_reference['International plan'], international_plan_reference['International plan_Code']))
        voice_mail_plan_mapping = dict(zip(voice_mail_plan_reference['Voice mail plan'], voice_mail_plan_reference['Voice mail plan_Code']))

        # Map categorical variables
        input_data['State'] = input_data['State'].map(state_mapping)
        input_data['International plan'] = input_data['International plan'].map(international_plan_mapping)
        input_data['Voice mail plan'] = input_data['Voice mail plan'].map(voice_mail_plan_mapping)

        # Check for missing values after mapping
        if input_data.isnull().values.any():
            logger.warning("Missing values detected after mapping: %s", input_data.isnull().sum())
            raise HTTPException(status_code=400, detail="Certain values could not be mapped. Please check your input data.")

        # Standardize numerical columns
        input_data[numerical_cols] = scaler_loaded.transform(input_data[numerical_cols])
        logger.info("Numerical columns standardized.")

        # Make predictions with both models
        pred_decision_tree = model_decision_tree.predict(input_data)
        pred_random_forest = model_random_forest.predict(input_data)
        logger.info("Predictions made successfully. Decision Tree: %s, Random Forest: %s", pred_decision_tree[0], pred_random_forest[0])

        # Return the results
        return {
            "Prediction_Decision_Tree": int(pred_decision_tree[0]),
            "Prediction_Random_Forest": int(pred_random_forest[0]),
        }
    except Exception as e:
        logger.error("Error during prediction: %s", e)
        raise HTTPException(status_code=500, detail="An error occurred during prediction.")
