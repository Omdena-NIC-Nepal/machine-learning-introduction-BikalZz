### Project on Introduction to Maching Learning

The core objective of this project is to introduce students to supervised learning, focusing on linear regression, by guiding them through a project that predicts house prices based on a variety of features.

---

#### Step 1: Setting up the environment
- **The project structure** has been set up as provided on the README.md of this project provided below:

```    .
    ├── data/
    │   ├── boston_housing.csv      : Dataset used in this project
    |   ├── download_dataset.ipynb  : Getting the dataset for analysis (added)
    |   ├── processed_data.pkl      : Datasets saved after Data preprocessing (added)
    |   ├── feature_importance.csv  : Saving the feature importance of the model
    |   └── predictions.csv         : Saving the final predictions of the model with feature engineering
    |
    ├── models/                         :(added)
    │   ├── best_model.pkl              : Best model selected after hyperparameter tuning (added)
    │   ├── feature_engineering.pkl     : Feature engineering pipline (added)
    │   ├── metrics.pkl                 : model performance evaluation metrics (added)
    |
    ├── notebooks/
    │   ├── EDA.ipynb                   : Data Exploration and Preprocessing (Finding pattern and issues)
    │   ├── Data_Preprocessing.ipynb    : Handling missing data (cleaning and preparing the data)
    │   ├── Model_Training.ipynb        : Choosing machine learing techniques and training them (eg. fit regression technique to predict prices)
    │   ├── Model_Evaluation.ipynb      : Measuring the accuracy of the model an ensure the model works well***
    │   └── Feature_Engineering.ipynb   : Use feature engineering techniques to improve the model*** (added)
    |
    ├── scripts/
    │   ├── data_preprocessing.py
    │   ├── train_model.py
    │   ├── evaluate_model.py
    |   └── feature_engineering.py      : (added)
    |
    ├── README.md                      : Provides the context of the assignment
    ├── requirements.txt
    ├── Project_Documentation.md                : Describes the project basic approach for this machine learning project
    └── .gitignore
```

- The **python version 3.12.7** is used for this project.

---

#### Step 2: Libraries/Modules/Dependencies
- **kagglehub**         : *For downloading the dataset*
- **numpy**             : *For numeric analysis*
- **pandas**            : *For data manipulation*
- **scikit-learn**      : *For machine learning*
- **matplotlib**        : *For data visualization*
- **seaborn**           : *For enhanced data visualization*
- **os**                : *For handling the path directory*
These libraries can be installed by writing the following code in the terminal: " pip install -r requirements.txt"

---

#### Step 3: Getting the dataset
- **Dataset:** The dataset **boston_housing.csv** has been download from kagglehub(altavish/boston-housing-dataset) using its library as per the code provided on download_dataset.ipynb file on data folder and copyed to the data folder of this project.

**Features:**

- **CRIM**: per capita crime rate by town.
- **ZN**: proportion of residential land zoned for lots over 25,000 sq. ft.
- **INDUS**: proportion of non-retail business acres per town.
- **CHAS**: Charles River dummy variable (1 if tract bounds river; 0 otherwise).
- **NOX**: nitrogen oxides concentration (parts per 10 million).
- **RM**: average number of rooms per dwelling.
- **AGE**: proportion of owner-occupied units built prior to 1940.
- **DIS**: weighted distances to five Boston employment centers.
- **RAD**: index of accessibility to radial highways.
- **TAX**: full-value property tax rate per $10,000.
- **PTRATIO**: pupil-teacher ratio by town.
- **B**: 1000(Bk - 0.63)^2 where Bk is the proportion of black residents by town.
- **LSTAT**: percentage of lower status of the population.
- **MEDV**: median value of owner-occupied homes in $1000s (Target).
---

#### Step 4: EDA (Exploratary Data Analysis)
1. Loaded the dataset from "..data/boston_housing.csv"
2. Explored Data structure and Summary statistic:
    - **Sample Data:** 506 entries
    - **Data Columns:** 14
    - **Data Types:** int64, float64
    - Found missing values which shall be handled.
3. Feature relation to target variable defined by this assignment(MEDV):
    - Out of 13 features 6 features are positively correlated and 7 features are negatively correlated with target variable(MEDV).
    - Top 5 correlated features orderly from high to low are: LSTAT, RM, PTRATIO, INDUS, TAX of which only RM is positively correlated and all others are negatively correlated.
4. Visual Plot:
    - Outliers are present on the datasets so these has to be handeled properly.
    - ZN, CRIM, B, MEDV has extreme outliers
5. Missing Values:
    - There are 20 missing values in each 6 features: CRIM, Z, INDUS, CHAS, AGE, LSTAT which are scattered in the datasets so dumping the rows of dataset is not preferable as we might loose around 20*6=120 number of datasets so impute method is preferred.
6. All the datasets/ sample are numerical datatype while some can be categorized on categorical data such as CHAS, RAD.
---

#### Step 5: Data Preprocessing
1. Missing values were handled using **SimpleImputer** from sklearn.impute with **median** strategy.
2. Outliers are handled using Interquartile Range(IQR rule) which is widely used statical technique for handling outliers.
    - **Quartiles:**
        - $$Q1$$ (25th percentile): Median of first half of data
        - $$Q2$$(75th percentile): Median of 2nd half of data
        - **IQR** (InterQuartile Range) represents middle of 50% of data:

            $
            IQR = Q2 - Q1
            $

    - **Quartile Thresholds:**
        - Lower Bound: Values below this are considered outliers.

            $
            LB = Q1 - 1.5 * IQR
            $

        - Upper Bound: values above this are considered outliers.

            $
            UB = Q2 + 1.5 * IQR
            $
            
    - **Replacing Outliers**
        - Any value greater than upper bound is replaced with the upper bound.
        - Any value lowere than lower bound is replaced with the lower bound.
        - Values within the bounds remain unchanged.
3. **Charles River dummy variable (CHAS)** value is in 1 and 0 so its datatype is converted to category datatype using **pandas.astype()** function. 
4. **Index of accessibility to radial highways (RAD)** is hot encoded as categorical variables using **pandas.get_dummies()** function which provides us with the **boolean dataype**.
5. All the int64 and float64 datatype datasets are normalized/standarized using **StandardScaler** method from **sklearn.preprocessing** so that the difference in different feature values magnitude wont influence the model and every features will have a mean of 0 and a standard deviation of 1.
6. Then, the datasets are split into training and testing sets with the training size of 80% of the total datasets and 20% for the testing datasets. 
    - **X_train:** Refers to the 80% of the training feature datasets for training the model.
    - **X_test:** Refers to the 20% of the testing feature datasets used for predicting target variable by the model.
    - **y_train:** Refers to the 80% of the target varialble datasets for trainig the model.
    - **y_test:** Refers to the 20% of the target variable datasets for testing and evaluating the predicted target variable of the model.
7. The datas X_train, X_test, y_train, y_test (processed data) with scaler is saved to the data folder with the filename **processed_data.pkl** using **joblib** module for training the model in future. 
3. **data_preprocessing.py script** is updated with the notebook code.
---

#### Step 6: Training the model
1. **Loading Dataset:** 
    - X_train, X_test, y_train, y_test datasets are retrived from the previously saved file(**processed_data.pkl**) using **joblib** module.
    - Top 5 features from EDA **(LSTAT, RM, PTRATIO, INDUS, TAX)** and two categorical variable from Data preprocessing **(RAD_4.0 and CHAS)** are selected as top features to train the model.
2. **Training the model:**
    - Linear regression algorithm is used to train the datasets.
    - **LinearRegressoin** method is used from the **sklearn.linear_model** with **fit()** function using X_train and y_train datasets to train the model.
    - **predict** function from the **LinearRegression** method is used to predict the target variable after the training of the model.
    - **(RMSE) Root Mean Square Error** and **R²_score** metrics are used from **sklearn.metrics** to evaluate the performance of the model. The performance achieved is as below:
        - RMSE: 3.72
        - R² Score: 0.72
3. **Hyperparameter tuning**
    - Hyperparameter tuning is performed on the same dataset using **Ridge Regression, Random Forest Regressor** and **XGBoost Regressor** algorithm.
    - **Ridge Regression:** **Ridge** method from **sklearn.linear_model** with **fit()** function using X_train and y_train datasets to train the model.
    - **Random Forest Regressor:** **RandomForestRegressor** method from **sklearn.ensemble** with **fit()** function using X_train and y_train datasets to train the model.
    - **XGBoost Regressor:** **XGBRegressor** method from **xgboost** with **fit()** function using X_train and y_train datasets to train the model.
    - **Grid Search Cross Validation**, a hyperparameter tuning technique is used in conjunction with the machine learning algorithms mentioned above for hyperparameter tuning.
        - **GridSearchCV** method from **sklear.model_selecton** is used to split the data into k fold and select the best estimator and parameter from the k fold after training the model.
    - The performance of the Hyperparameter tuning are provided below:
        | Model                   | RMSE                        | R² Score               | Best_Params                                                  |
        |:------------------------|:----------------------------|:-----------------------|:-------------------------------------------------------------|
        | XGBoost                 | 3.032855                    | 0.811994               | '{'gamma': 0, 'learning_rate': 0.1, 'max_depth'...}'         |
        | RandomForest            | 3.180640                    | 0.793225               | '{'max_depth': 10, 'min_samples_leaf': 2, 'min_...}'         |
        | Ridge                   | 3.711583                    | 0.718429               | '{'alpha': 10}'                                              |

4. **Selecting and Saving the model:**
    - From the performance of above we can see that the best performed model is XGBoost which is cost and time effective. Therefore, XGBoost model is selected and saved in **model** folder with filename **best_model.pkl** for future use.
5. **train_model.py** script file is updated with the notebook code.
---

#### Step 7: Evaluating the Model
1. Processed data and best model is loaded from the data and models folder repectively using **joblib**.
2. Top features and categorical features mentioned in step 6 were selected for testing.
3. The target variable is predicted using X_test features using **predict()** function of the XGBRegressor method.
4. RMSE, MSE (Mean Squared Error), R² Score and MAE (Mean Absolute Error) is used as metrics for Model Evaluation from **sklearn.metrics**.
5. The performance is as follows:
    | **RMSE**                   | **MSE**	                       | **R² Score**               | **MAE**                   |
    |:-----------------------|:----------------------------|:-----------------------|:----------------------|
    | 3.032855               | 9.198207                    | 0.811994               | 2.217479              |
6. The sample predicted values are:
    | **Real**                  | **MSE**	                       | 
    |:-----------------------|:----------------------------|
    | 24.0                   | 27.013309                   |
    | 34.7                   | 34.947609                   |
    | 18.9                   | 19.348469                   |
    | 18.9                   | 20.947292                   |
    | 20.2                   | 18.723349                   |

7. **Visual Plot**
    - Residuals vs Predicted Values Plot  
        - **Observations**:  
            - Residuals range from **0.0 to 7.5**, with most clustered between **0.0–5.0**.  
            - More clustered at mid range price with high frequency.  

    - Distribution of Residuals (Histogram)  
        - **Observations**:  
            - Roughly symmetric around **0**, but slight right skew.  
            - Most residuals fall within **±5.0**, with fewer extreme values.  
8. **Feature Importance:**
    - **hasattr()** a python builtin function is used to check if the feature importance exist. If exist the bar plot is shown showing the importance of features.
    - From the bar plot it is observed that **LSTAT** and **RM** are the most influencial features that dominates in the prediciton of the model.
9. **Saving metrics, feature importance and prediction:**
    - Evaluated metrics is saved in the models folder with filename **metrics.pkl** using **joblib**.
    - Feature importance data is saved in the data folder with filename **feature_importance.csv** using **pandas.to_csv()** function.
    - Model predictions data is saved in the data folder with filename **predictions.csv** using **pandas.to_csv()** function.
10. **evaluate_model.py** script file is updated with the notebook code as a production ready file.
---

#### Step 8: Feature Engineering
1. Processed data and model is loaded from data and models folder using **joblib** module.
2. Model and best parameter is extracted from the loaded model.
3. Train and test features are extracted from the processed data for the selected features.
4. New features are engineered for **X_train** and **X_test**:
    - Rooms per tax
        $$
        \text{ROOMS\_PER\_TAX} = \frac{RM}{TAX + 10^{-6}}
        $$
    - LSTAT and RM interaction
        $$
        \text{LSTAT\_RM\_INTERACTION} = LSTAT \times RM
        $$
    - LSTAT squared
        $$
        \text{LSTAT\_SQUARED} = LSTAT^2
        $$
    - RM binned into categorical ranges.

5. Model is retrained with new engineered features with proper early stopping to prevent overfitting and saves computation time. Target variable prediction is done after training the model and the performance of the model is evaluated using **root mean square error** metrics.
6. Feature importance analysis with selected features and engineered features is performed where **RM**, **LSTAT** and **TAX** are the top 3 important features observed.
7. Top features based on feature importance is selected using **SelectFromModel** method from **sklearn.feature_selection** with *median* threshold. The output of the top important features from feature selection are **RM, LSTAT, PTRATIO, TAX, LSTAT_RM_INTERACTION, LSTAT_SQUARED**.
8. Model is retrained with top selected engineered features and prediction on target variable is provided.
9. Final predictions are save to '..data/predictions.csv' and final feature importance is saved to '../data/feature_importance.csv'.
10. **feature_engineering.py** files is updated with the notebook code as a production ready file. Also, the whole pipeline for feature engineering is saved to '../models/feature_engineering.pkl'.
---

### **Brief Summary of the Machine Learning Project**

This project introduces **supervised learning** (linear regression) to predict Boston house prices (**MEDV**) using key features like room count, and tax rates. Here’s a brief breakdown:

#### **1. Key Steps & Outcomes**  
- **Data Preparation**:  
  - Handled missing values (`SimpleImputer`) and outliers (IQR method).  
  - Encoded categorical variables (`CHAS`, `RAD`) and standardized features.  
  - Split data: **80% train**, **20% test**.  

- **Model Training**:  
  - **Baseline (Linear Regression)**: RMSE: 3.72, R²: 0.72.  
  - **Hyperparameter Tuning**:  
    - **XGBoost** outperformed Ridge and Random Forest (RMSE: **3.03**, R²: **0.81**).  
    - Best model saved (`best_model.pkl`).  

- **Feature Engineering**:  
  - Created new features (e.g., `ROOMS_PER_TAX`, `LSTAT_RM_INTERACTION`).  
  - Retrained XGBoost with improved feature importance (eg: **RM, LSTAT, TAX**).  

#### **2. Performance Metrics**  
| Model          | RMSE   | R²     |  
|----------------|--------|--------|  
| Linear Regression | 3.72   | 0.72   |  
| **XGBoost (Tuned)** | **3.03** | **0.81** |  

#### **3. Insights**  
- **Top Features**: `LSTAT` (economic status) and `RM` (room count) drove predictions.  
- **Visualizations**: Residual plots confirmed model robustness (errors clustered near zero).  
- **Feature Engineering**: Improved interpretability and model performance.  

---
$
THANK YOU
$
---