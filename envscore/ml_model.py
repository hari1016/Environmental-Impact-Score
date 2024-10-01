import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import os
from sklearn.preprocessing import MinMaxScaler
from openai import OpenAI
import hiplot as hip
import json
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataset_path = os.path.join(BASE_DIR, 'dataset', 'food-footprints.csv')
model_path = os.path.join(BASE_DIR, 'models', 'kmeans_model.pkl')

df = pd.read_csv(dataset_path)
filtered_df = df.iloc[:,[0,4,8,12,16]]

columns = filtered_df.columns
scaler = MinMaxScaler()
normalized_df = scaler.fit_transform(filtered_df)
normalized_df = pd.DataFrame(normalized_df, columns=columns)

if os.path.exists(model_path):
    # Load the saved model
    Kmean = joblib.load(model_path)
else:
    Kmean = KMeans(n_clusters=3, random_state=42)
    Kmean.fit(normalized_df)
    os.makedirs(os.path.join(BASE_DIR, 'models'), exist_ok=True)
    joblib.dump(Kmean, model_path)

labels = Kmean.labels_

api_key = os.getenv('OPENAI_API_KEY')

client = OpenAI()


def predict_score(product_name):
    completion = client.chat.completions.create(
        model="ft:gpt-4o-mini-2024-07-18:personal::AD2OUOjZ",
        messages=[
            {"role": "system",
            "content": """ For a given food product, provide the Emissions per kilogram(kilograms CO2), Water withdrawals per kilogram(liters), Land use per kilogram(meter square), Water scarcity per kilogram(liters) and Eutrophication per kilogram(gPO4) of that product in JSON format.
            For a complex food product, calculate these attributes by averaging the values for each of the ingredients used to make that food product."""},
            {"role": "user", "content": product_name}
        ]
    )

    print(completion.choices[0].message.content)
    output = completion.choices[0].message.content
    output_dict = json.loads(output)
    emissions = output_dict["Emissions per kilogram"]
    land_use = output_dict["Land use per kilogram"]
    eutrophication = output_dict["Eutrophication per kilogram"]
    water_scarcity = output_dict["Water scarcity per kilogram"]
    water_withdrawals = output_dict["Water withdrawals per kilogram"]

    # Convert the values into a NumPy array and reshape it
    input_df = np.array([emissions, land_use, eutrophication, water_scarcity, water_withdrawals]).reshape(1, -1)
    scaled_input = scaler.transform(input_df)
    prediction = Kmean.predict(scaled_input)
    return prediction[0]

