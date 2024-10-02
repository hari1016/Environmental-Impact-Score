# Environmental Impact Score for Products

## Overview
This project aims to provide consumers with a clear understanding of the environmental impact of products by giving each product a score based on factors such as carbon emissions, water usage, and waste generated throughout its lifecycle. The goal is to empower individuals to make eco-conscious purchasing decisions.

## Features
- **Environmental Impact Score**: Get a score for any product by entering its name. The score is calculated using various environmental factors.
- **Custom Machine Learning Model**: The core engine behind the score is a fine-tuned OpenAI model trained on environmental data specific to various products and industries.
- **User-Friendly Interface**: A simple web interface where users can search for products and receive their environmental impact score.
- **Sustainable Choices**: The platform promotes environmentally responsible consumerism by providing valuable insights into the sustainability of products.

## Inspiration
We developed this platform to help consumers make informed choices in their daily lives by evaluating the environmental costs associated with everyday products.

## How It Works

1. **Data Collection**: 
   - Raw data is collected from publicly available sources such as [Our World in Data](https://ourworldindata.org/), which provides a wide range of environmental metrics. For experimental purposes, OpenAI’s API was also used to gather supplementary data.
   
2. **Data Processing**: 
   - The raw data undergoes cleansing and transformation using Python libraries such as NumPy, Pandas, and Scikit-learn. This ensures that the data is consistent, clean, and ready for further analysis.

3. **Feature Engineering**: 
   - Specific columns are filtered from the dataset, focusing on critical environmental metrics such as greenhouse gas emissions, water usage, land use, water scarcity, and eutrophication values. This step highlights the most relevant data points for score calculation.

4. **Normalization**: 
   - To ensure that each feature contributes equally to the model's calculations, the selected data is normalized using MinMaxScaler. Normalization is essential for clustering algorithms like K-means, as these are sensitive to the scale of data. This process scales the data into a uniform range, optimizing the performance of our algorithms.

5. **Model Training**: 
   - The processed data is then fitted into a K-means clustering algorithm with the number of clusters (k) set to 3. This clustering allows us to categorize products based on their environmental impact, grouping them into distinct clusters for better analysis.

6. **Cluster Analysis and Score Calculation**: 
   - Using HiPlot for cluster visualization, the clusters are analyzed and ranked based on their environmental behaviors. These rankings are used to calculate the *Environmental Impact Score (EIS)*, providing a clear metric for consumers.

7. **LLM Fine-Tuning**: 
   - The processed environmental data is transformed into a conversational format. This dataset is used to fine-tune OpenAI's `gpt-4o-mini-2024-07-18` model, allowing users to interact with the model in a conversational manner, receiving insights on product sustainability.

8. **Data Presentation**: 
   - The calculated Environmental Impact Scores are retrieved by the Django server and displayed to the user via an interactive and animated slider on the website. This interface provides a seamless experience, allowing users to easily understand and engage with the product’s environmental footprint.

## Future Plans
   - Expanded Dataset: Add more products and industries to provide a broader range of environmental impact scores.
   - Recommendation Engine: Suggest eco-friendly alternatives based on the score.
