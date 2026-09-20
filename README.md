# Comparative Analysis of Sequential Models for Mobile Network Traffic Forecasting

## Project Overview

This project investigates one-step-ahead mobile Internet traffic forecasting using sequential deep learning models on the Milan telecommunications activity dataset.

The main research question is:

> How do different sequential models compare for one-step-ahead mobile network traffic forecasting, and how does their performance vary across geographical areas with different traffic characteristics?

The study focuses on Internet traffic recorded across geographical areas in Milan at 10-minute intervals. Three neural network architectures are implemented and compared using the same forecasting task, chronological data split, preprocessing procedure, and evaluation period.

The project was developed and executed primarily in Google Colab, with the project files and outputs organized for reproducibility through this GitHub repository.

---

## Dataset

The project uses the **Telecommunications - SMS, Call, Internet - MI** dataset for Milan from Barlacchi et al. (2015).

The dataset contains mobile network activity collected across approximately 10,000 geographical areas ("squares") in Milan. Observations are recorded at 10-minute intervals and include:

- Square/geographical area ID
- Timestamp
- Country code
- Incoming SMS activity
- Outgoing SMS activity
- Incoming call activity
- Outgoing call activity
- Internet traffic

Only the **Internet traffic** variable is used for the forecasting experiments in this project.

### Dataset Source

The dataset is available through Harvard Dataverse:

https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/EGZHFV

### Original Dataset Reference

Barlacchi, G., De Nadai, M., Larcher, R., Casagrande, G., Vanhoof, M., Cattuto, C., Lepri, B., and Antonelli, F. (2015).

*A multi-source dataset of urban life in the city of Milan and the Province of Trentino.*

Scientific Data, 2, 150055.

DOI: https://doi.org/10.1038/sdata.2015.55

---

# Repository Structure

```text
mobile_traffic_forecasting/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   └── mobile_traffic_forecasting.ipynb
│
├── data/
│   └── README.md
│
├── figures/
│   ├── total_internet_traffic_distribution.png
│   ├── square_5161_first_two_weeks.png
│   ├── square_5059_first_two_weeks.png
│   ├── square_5259_first_two_weeks.png
│   ├── square_4159_first_two_weeks.png
│   ├── square_4556_first_two_weeks.png
│   ├── square_5161_acf.png
│   ├── square_5161_hourly_seasonality.png
│   └── ...
│
├── results/
│   ├── final_performance_square_5161.csv
│   ├── final_performance_square_5059.csv
│   ├── final_performance_square_5259.csv
│   ├── overall_comparison.csv
│   ├── cross_area_table.csv
│   ├── report_characteristics.csv
│   └── failure_report.csv
│
└── models/
    └── ...
Directory descriptions

notebooks/

Contains the complete Google Colab notebook used for data processing, exploratory analysis, model development, training, evaluation, visualization, and result generation.

data/

Contains documentation describing the dataset and data organization. The large raw dataset files are not included in the GitHub repository.

figures/

Contains the figures generated during exploratory analysis and forecasting evaluation.

results/

Contains the numerical outputs used for model comparison, cross-area analysis, model characteristics, and prediction-error analysis.

models/

Contains model-related project files where applicable.

Data Availability and Resource Constraint

The original Milan dataset is very large, with the raw tab-separated files requiring substantial storage.

Because of available storage and computational resources, this project retained two specific observation windows:

1–14 November 2013 — exploratory analysis and model training
16–22 December 2013 — final forecasting evaluation

The raw files from these retained periods were processed using memory-conscious techniques.

The complete original dataset was therefore not downloaded and processed in its entirety for this project.

Consequently, geographical-area rankings reported in this study represent the retained observation period rather than the complete original two-month dataset.

This limitation is explicitly considered when interpreting the exploratory results.

The raw dataset is not included in this repository because of its large size.

Data Handling and Memory Management

The original files contain multiple activity variables and multiple country-code records for geographical areas and timestamps.

To make processing feasible within the available computational resources, the project uses several memory-management techniques.

Chunked Processing

Large raw files are read in chunks rather than loading an entire file into memory at once.

The processing uses a chunk size of:

100,000 rows

This allows aggregation to be performed incrementally.

Column Selection

Only the columns required for the Internet traffic analysis are loaded during the relevant processing stages:

square_id
timestamp
internet

Unnecessary activity variables are not loaded when they are not required for a particular analysis.

Optimized Data Types

The following data types are used where appropriate:

optimized_dtypes = {
    "square_id": "int32",
    "timestamp": "int64",
    "country_code": "int32",
    "sms_in": "float32",
    "sms_out": "float32",
    "call_in": "float32",
    "call_out": "float32",
    "internet": "float32"
}

Using float32 and int32 where appropriate reduces memory consumption compared with default pandas data types.

Aggregation

The raw data can contain multiple country-code records for the same geographical area and timestamp.

Internet traffic is therefore aggregated by:

square_id + timestamp

using the sum of Internet activity.

This produces one Internet traffic observation for each geographical area and timestamp.

Memory Optimization Result

During the memory experiment on a representative raw file, the estimated DataFrame memory usage was approximately:

Before optimization: 0.29 GB
After optimization:  0.16 GB
Reduction:            43.75%

The experiment demonstrates why data-type optimization and selective loading are important when working with the original dataset.

Exploratory Data Analysis

The exploratory analysis investigates the characteristics of mobile Internet traffic before model selection.

The analysis includes:

Distribution of total Internet traffic across geographical areas
Identification of high-traffic geographical areas
Time-series analysis during the first two weeks
Comparison of high-traffic and specified geographical areas
Autocorrelation analysis
Hourly/seasonal traffic analysis
Geographical Areas

Based on the retained observation period, the three highest-traffic geographical areas identified for the forecasting experiments are:

5161
5059
5259

The analysis also includes the assignment-specified areas:

4159
4556

Therefore, five areas are examined during the exploratory analysis:

5161
5059
5259
4159
4556

The three highest-traffic areas are subsequently used for the cross-area forecasting comparison.

Forecasting Task

The forecasting problem is formulated as one-step-ahead prediction.

For each prediction:

Previous observed Internet traffic values are provided to the model.
The model predicts the next Internet traffic value.
The prediction is compared with the actual observed value.

The data is sampled every 10 minutes.

Sequence Construction

A sequence length of 24 observations is used.

Because observations occur every 10 minutes:

24 observations × 10 minutes = 240 minutes

Therefore, each input sequence represents the previous 4 hours of Internet traffic.

The forecasting task can be represented as:

[t-23, t-22, ..., t-2, t-1] → predict t

where each value represents Internet traffic for a geographical area at a 10-minute interval.

Using a fixed historical window allows the models to learn short-term temporal dependencies while keeping the input size manageable.

Data Splitting

The data is split chronologically to prevent future observations from being used to predict earlier observations.

The forecasting experiment uses:

Training:
November 2013 retained observation period

Validation:
Chronological validation portion used during model development

Testing:
16–22 December 2013

The evaluation week is fixed in advance and is not used to select the models.

Chronological splitting is important for time-series forecasting because randomly shuffling observations could introduce information from the future into the training data.

Data Preprocessing
Normalization

Internet traffic values are normalized using MinMax scaling.

The scaler is fitted using the training data only.

The learned transformation is then applied to the validation and test data.

This prevents information from the evaluation period from influencing the scaling parameters.

The general transformation is:

x_scaled = (x - x_min) / (x_max - x_min)

After prediction, the model outputs are transformed back to the original Internet traffic scale for evaluation.

Models

Three different neural-network architectures are compared for the one-step-ahead forecasting task.

All models receive the same sequence representation and use the same general preprocessing and chronological evaluation procedure.

1. Multi-Layer Perceptron (MLP)

The MLP treats the 24-step input sequence as a fixed-size feature vector.

The baseline architecture contains:

Input: 24 × 1
      ↓
Flatten
      ↓
Dense(64, ReLU)
      ↓
Dense(32, ReLU)
      ↓
Dense(1)

The MLP provides a non-recurrent neural baseline for comparison.

It can learn nonlinear relationships among the values in the input window but does not explicitly maintain a recurrent hidden state.

2. One-Dimensional Convolutional Neural Network (1D CNN)

The CNN applies one-dimensional convolution across the temporal sequence.

The baseline architecture contains:

Input: 24 × 1
      ↓
Conv1D(32, kernel_size=3, ReLU)
      ↓
MaxPooling
      ↓
Flatten
      ↓
Dense(32, ReLU)
      ↓
Dense(1)

The convolutional layer allows the model to learn local temporal patterns in the recent traffic sequence.

This makes the CNN useful for detecting short-term patterns and local changes in traffic.

3. Long Short-Term Memory Network (LSTM)

The LSTM is a recurrent neural network designed to model sequential information.

The baseline architecture contains:

Input: 24 × 1
      ↓
LSTM(64)
      ↓
Dense(32, ReLU)
      ↓
Dense(1)

The LSTM maintains a recurrent hidden representation of the input sequence, allowing it to model temporal dependencies in the traffic observations.

Model Training

The models are trained using the training portion of the retained November data.

Validation data is used to monitor model performance during training.

Early stopping is used during model development to reduce unnecessary training once validation performance stops improving.

The model selection process considers both predictive performance and the characteristics of the traffic series identified during exploratory analysis.

The final evaluation is performed on the fixed period:

16–22 December 2013
Evaluation Areas

The final forecasting experiments are performed across the three highest-traffic geographical areas identified from the retained observation period:

Square 5161
Square 5059
Square 5259

Each model is evaluated separately for each area.

This produces:

3 models × 3 geographical areas = 9 actual-vs-predicted plots

The repository contains these plots in the figures/ directory.

Evaluation Metrics

The models are evaluated using multiple error metrics.

Mean Absolute Error (MAE)

MAE measures the average absolute difference between the actual and predicted traffic values.

MAE = mean(|actual - predicted|)

Lower MAE indicates smaller average prediction errors.

Root Mean Squared Error (RMSE)

RMSE gives greater weight to larger prediction errors.

RMSE = sqrt(mean((actual - predicted)^2))

This is useful for identifying whether a model produces particularly large forecasting errors during traffic spikes.

Mean Absolute Percentage Error (MAPE)

MAPE measures prediction error relative to the actual traffic level.

Zero-valued actual observations are handled separately so that division by zero does not distort the metric.

The final metric tables are stored in the results/ directory.

Results

The repository contains model-comparison tables for:

Square 5161
Square 5059
Square 5259

The main result files include:

results/final_performance_square_5161.csv
results/final_performance_square_5059.csv
results/final_performance_square_5259.csv
results/overall_comparison.csv
results/cross_area_table.csv

These files contain the quantitative results used to compare the models.

The forecasting figures show the actual and predicted Internet traffic during the evaluation period from 16–22 December 2013.

Error Analysis

The project also examines prediction errors rather than relying only on aggregate metrics.

The error analysis considers:

Absolute prediction error
Underprediction
Periods with unusually large errors
Differences in model behavior during traffic changes

A separate failure analysis is stored in:

results/failure_report.csv

This analysis is used to identify periods where the models have difficulty following unusual or rapidly changing traffic behavior.

Temporal Analysis

Two additional analyses are performed on the highest-traffic area.

Autocorrelation

Autocorrelation is examined using the Internet traffic series for Square 5161.

The analysis evaluates how strongly traffic at one time is related to previous observations.

For example, the observed autocorrelation values for the first day include:

Lag 1  = 0.9835
Lag 6  = 0.9180
Lag 12 = 0.7610
Lag 24 = 0.3373

The strong short-lag autocorrelation indicates substantial temporal dependence in the traffic series.

Because each observation represents 10 minutes:

Lag 1  = 10 minutes
Lag 6  = 1 hour
Lag 12 = 2 hours
Lag 24 = 4 hours

This provides evidence that recent traffic observations contain useful information for forecasting the next observation.

Hourly Seasonality

Hourly traffic patterns are also examined to identify recurring temporal behavior across the day.

This analysis helps determine whether traffic levels vary systematically with time of day and provides additional context for interpreting forecasting performance.

Figures

The figures/ directory contains the visual outputs generated during the study.

Important figures include:

total_internet_traffic_distribution.png

Distribution of total Internet traffic across geographical areas.

square_5161_first_two_weeks.png
square_5059_first_two_weeks.png
square_5259_first_two_weeks.png
square_4159_first_two_weeks.png
square_4556_first_two_weeks.png

First-two-week Internet traffic time series for the five areas examined in the exploratory analysis.

square_5161_acf.png

Autocorrelation analysis for the highest-traffic area.

square_5161_hourly_seasonality.png

Hourly/seasonal traffic analysis for the highest-traffic area.

The directory also contains the nine actual-versus-predicted plots for the three models across the three evaluation areas.

How to Run the Project
1. Clone the Repository

Clone this repository to your computer:

git clone <YOUR_GITHUB_REPOSITORY_URL>

Move into the project directory:

cd mobile_traffic_forecasting
2. Create a Python Virtual Environment

Create a virtual environment:

python -m venv .venv

Activate it on Windows Git Bash:

source .venv/Scripts/activate

If using Windows Command Prompt instead:

.venv\Scripts\activate
3. Install Dependencies

Install the required Python packages:

pip install -r requirements.txt

The required packages are listed in:

requirements.txt
Running the Notebook

The complete project implementation is contained in:

notebooks/mobile_traffic_forecasting.ipynb

The notebook was developed and executed using Google Colab.

To reproduce the analysis:

Clone or download this repository.
Open Google Colab.
Upload/open mobile_traffic_forecasting.ipynb.
Mount Google Drive when prompted by the notebook.
Ensure the required project data is available at the expected Google Drive location.
Install any required dependencies listed in requirements.txt.
Run the notebook cells in order.

The notebook contains the complete workflow for:

Data loading
    ↓
Memory optimization
    ↓
Data aggregation
    ↓
Exploratory analysis
    ↓
Temporal analysis
    ↓
Sequence construction
    ↓
Preprocessing/scaling
    ↓
Model training
    ↓
Hyperparameter/model development
    ↓
Evaluation
    ↓
Visualization
    ↓
Error analysis
    ↓
Result tables
Data Setup

The raw dataset is intentionally not included in this GitHub repository because of its large size.

To reproduce the raw-data processing, obtain the dataset from the official Harvard Dataverse source:

https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/EGZHFV

The retained project files correspond to:

1–14 November 2013
16–22 December 2013

The notebook should be configured to point to the location where the retained raw files or processed data are stored.

Reproducibility

The project is designed so that the main analysis can be reproduced from the notebook and the files provided in this repository.

The notebook records the major steps used in the study, including:

Data processing
Memory optimization
Geographical-area aggregation
Exploratory analysis
Sequence generation
Normalization
Model construction
Model training
Evaluation
Visualization
Error analysis

The saved figures and CSV results in this repository provide the outputs used in the final analysis.

Because the complete original dataset was not retained locally, reproducing the exact complete-dataset ranking would require access to the full original dataset.

Computational Environment

The project was developed using:

Python
Google Colab
TensorFlow/Keras
NumPy
Pandas
Matplotlib
Seaborn
Scikit-learn

Model training was performed using the Google Colab computational environment.

The exact package dependencies required by the project are provided in:

requirements.txt
Project Limitations

Several limitations should be considered when interpreting the results.

Limited retained observation period

The complete original dataset was not retained because of storage and computational constraints.

The analysis therefore uses November 1–14 and December 16–22 rather than the complete original observation period.

As a result, the geographical-area ranking represents the retained observation period.

Evaluation period

The final evaluation is limited to December 16–22, 2013, as required by the forecasting task.

Performance during another period may differ.

Geographical coverage

The forecasting comparison focuses on three high-traffic geographical areas.

The behavior of other areas may differ because geographical areas can have different traffic characteristics.

Sequence length

A 24-observation input window represents four hours of historical traffic.

Although this captures recent temporal behavior, a longer sequence could potentially capture additional daily or longer-term dependencies.

Model scope

Only three neural architectures are compared.

Other forecasting approaches, including statistical models, transformer-based architectures, or other sequence models, could provide additional comparisons.

Future Improvements

Potential improvements to the project include:

Processing the complete original dataset when sufficient storage and computational resources are available.
Evaluating additional geographical areas with different traffic characteristics.
Testing longer input sequence lengths.
Incorporating additional temporal features such as hour of day and day of week.
Investigating additional recurrent and attention-based architectures.
Performing more extensive hyperparameter optimization.
Evaluating the models across additional time periods.
Investigating anomaly-aware forecasting approaches for unusual traffic spikes.
References

[1] G. Barlacchi et al., "A multi-source dataset of urban life in the city of Milan and the Province of Trentino," Scientific Data, vol. 2, article 150055, 2015.
DOI: https://doi.org/10.1038/sdata.2015.55

[2] Harvard Dataverse, "Telecommunications - SMS, Call, Internet - MI - A multi-source dataset of urban life in the city of Milan and the Province of Trentino Dataverse."
https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/EGZHFV

[3] F. Chollet, Deep Learning with Python. Manning Publications.

[4] TensorFlow Documentation, "Keras Sequential model."
https://www.tensorflow.org/guide/keras/sequential_model

[5] Scikit-learn Documentation, "MinMaxScaler."
https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html

Author

Hikma Hamza

BSc Software Engineering
African Leadership University

