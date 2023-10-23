import logging
import requests
import pickle

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

url = "http://0.0.0.0:5000/predict"
[encoder, lb, model] = pickle.load(open("artifact/models/lr_model.pkl", "rb"))
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

response = requests.post(
    url=url,
    json= {
        "age": 39,
        "workclass": "State-gov	",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 2174,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States"
    }
)
logger.info(f"Status code: {response.status_code}")
logger.info(response.json())
