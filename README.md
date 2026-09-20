# Comparative Analysis of Sequential Models for Mobile Network Traffic Forecasting

## Overview

This project investigates how different sequential and neural network models perform for one-step-ahead mobile network Internet traffic forecasting.

The study uses the Milan telecommunications dataset and compares three models:

- Multi-Layer Perceptron (MLP)
- Convolutional Neural Network (CNN)
- Long Short-Term Memory (LSTM)

The models are evaluated across geographical areas with different traffic characteristics.

## Research Question

How do different sequential models compare for one-step-ahead mobile network traffic forecasting, and how does their performance vary across geographical areas with different traffic characteristics?

## Project Structure

```text
mobile-traffic-forecasting/
├── data/
│   ├── raw/
│   └── processed/
├── figures/
├── models/
├── notebooks/
├── results/
├── .gitignore
├── README.md
└── requirements.txt

Dataset

The project uses the Milan telecommunications dataset containing SMS, call and Internet activity across geographical areas.

The original dataset is large, so the raw dataset files are not included in this GitHub repository.

The data used for this study was processed in Google Colab using Google Drive for storage.

Data Handling

The raw telecommunications files were processed in chunks to reduce memory usage.

The analysis retained:

1–14 November 2013 for exploratory analysis and model training
16–22 December 2013 for final model evaluation

Because of storage and computational resource limitations, the geographical-area ranking represents the retained observation period rather than the complete original dataset.

Models

Three different neural network architectures were implemented:

Multi-Layer Perceptron

The MLP uses the previous 24 observations as input and predicts the next Internet traffic value.

Convolutional Neural Network

The CNN uses one-dimensional convolution to extract local temporal patterns from the previous 24 observations.

Long Short-Term Memory

The LSTM processes the previous 24 observations sequentially and is designed to capture temporal dependencies.

Input Representation

Observations are recorded every 10 minutes.

A sequence length of 24 observations was used, representing the previous four hours of Internet traffic.

Min-Max normalization was applied, with the scaler fitted only on the training data.

Evaluation

The models are evaluated using:

Mean Absolute Error (MAE)
Root Mean Squared Error (RMSE)
Mean Absolute Percentage Error (MAPE)

Actual-versus-predicted plots are also provided for the evaluation period.