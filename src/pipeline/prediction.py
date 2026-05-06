import pickle
import pandas as pd

class CustomData:
    def __init__(self, city, furnishing_status, tenant_preferred,
                 area_type, bhk, size, bathrooms, floor_level):

        self.city = city
        self.furnishing_status = furnishing_status
        self.tenant_preferred = tenant_preferred
        self.area_type = area_type
        self.bhk = bhk
        self.size = size
        self.bathrooms = bathrooms
        self.floor_level = floor_level

    def get_data_as_data_frame(self):
        data_dict = {
            "City": [self.city],
            "Furnishing Status": [self.furnishing_status],
            "Tenant Preferred": [self.tenant_preferred],
            "Area Type": [self.area_type],
            "BHK": [self.bhk],
            "Size": [self.size],
            "Bathroom": [self.bathrooms],
            "Floor": [self.floor_level]
        }
        return pd.DataFrame(data_dict)


class PredictionPipeline:
    def __init__(self):
        self.model = pickle.load(open("model.pkl", "rb"))
        self.scaler = pickle.load(open("scaler.pkl", "rb"))
        self.columns = pickle.load(open("columns.pkl", "rb"))

    def predict(self, features):
        df = pd.get_dummies(features)
        df = df.reindex(columns=self.columns, fill_value=0)
        df_scaled = self.scaler.transform(df)

        return self.model.predict(df_scaled)