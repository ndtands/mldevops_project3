from typing import Dict
import pickle
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
import hydra

from core.data import process_data
from core.model import infer

api = FastAPI()

class CensusInputData(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(alias='education-num')
    marital_status: str = Field(alias='marital-status')
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias='capital-gain')
    capital_loss: int = Field(alias='capital-loss')
    hours_per_week: int = Field(alias='hours-per-week')
    native_country: str = Field(alias='native-country')

    class Config:
        schema_extra = {
            "examples": [
                {
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
            ]
        }


@api.get(path="/")
def welcome_root():
    return {"message": "Welcome to the tannd22 project!"}

@api.post(path="/predict")
async def prediction(input_data: CensusInputData) -> Dict[str, str]:
    """
    Post request for model predict
    Args:
        input_data (CensusInputData) : Instance of a CensusInputData object. Collected data from
        web form submission.
    Returns:
        dict: Dictionary containing the model output.
    """
    with hydra.initialize(config_path=".", version_base="1.2"):
        config = hydra.compose(config_name="configs")
    [encoder, lb, model] = pickle.load(
        open(config["main"]["model_path"], "rb"))
    input_df = pd.DataFrame(
        {k: v for k, v in input_data.dict(by_alias=True).items()}, index=[0]
    )

    processed_input_data, _, _, _ = process_data(
        X=input_df,
        categorical_features=config['main']['cat_features'],
        label=None,
        training=False,
        encoder=encoder,
        lb=lb
    )

    prediction = infer(model, processed_input_data)
    return {"Output": ">50K" if int(prediction[0]) == 1 else "<=50K"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app="api:api", host="0.0.0.0", port=5000)