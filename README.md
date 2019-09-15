# Kaggle-Elo-Merchant-Category-Recommendation
Kaggle competition: Elo Merchant Category Recommendation



## Summary:
- **Summary**: This is a regression problem based on simulated data. We're asked to predict customer loyalty based on each customer's historic transaction records

- **Challenge**: The targets are biased: There are 2207 (1.09%) outliers (target==-33.21928). Predicting them will be key of winnng this competition.

## Our Strategies
- **Feature engineering for train.csv and test.csv**: 

  - Create new datetime-related features: 
    'first_active_year', 'first_active_quarter', 'elapsed_time', 'days_feature1', 'days_feature2', 'days_feature3' etc.
  - One-hot-encoding for categorical features (feature_1, feature_2 & feature_3) (replace nan with mean values).
   
- **Feature engineering for historical_transactions.csv and new_merchant_transactions.csv **: 

  - fillna:
    ```
    df['merchant_id'].fillna('M_ID_00a6ca8a8a',inplace=True)
    df['category_2'].fillna(6.0,inplace=True)
    df['category_3'].fillna('D',inplace=True)
    ``` 
  - alphabets to numeric:
    ```
    df['authorized_flag'] = df['authorized_flag'].map({'Y': 1, 'N': 0}).astype(int)
    df['category_1'] = df['category_1'].map({'Y': 1, 'N': 0}).astype(int)
    df['category_3'] = df['category_3'].map({'A':1, 'B':2, 'C':3, 'D': 4}).astype(int)
    ```
  
  - Create new datetime features:
    ```
    df['purchase_date'] = pd.to_datetime(df['purchase_date'])
    df['purchase_quarter'] = df['purchase_date'].dt.quarter
    df['purchase_month'] = df['purchase_date'].dt.month
    df['purchase_day'] = df['purchase_date'].dt.day
    df['purchase_hour'] = df['purchase_date'].dt.hour
    df['purchase_weekofyear'] = df['purchase_date'].dt.weekofyear
    df['purchase_weekday'] = df['purchase_date'].dt.weekday
    df['purchase_weekend'] = (df['purchase_date'].dt.weekday >=5).astype(int)
    df['purchase_morning'] = (df['purchase_date'].dt.hour < 12).astype(int)
    ```
  - Create additoinal features:
    ```
    df['price'] = df['purchase_amount'] / df['installments']
    df['month_diff'] = ((datetime.datetime.today() - df['purchase_date']).dt.days)//30
    df['month_diff'] += df['month_lag']
    df['duration'] = df['purchase_amount']*df['month_diff']
    df['amount_month_ratio'] = df['purchase_amount']/df['month_diff']
    ```
    and many more ...
- **De-anonymizing data (This is considered an important step)** (Credit to [raddar](https://www.kaggle.com/raddar/towards-de-anonymizing-the-data-some-insights) ):
    ```
    for df in [hist_df, new_merchant_df]:
      df['purchase_amount'] = np.round(df['purchase_amount'] / 0.00150265118 + 498.06)
    merchants['numerical_1'] = np.round(merchants['numerical_1'] / 0.009914905 + 5.79639, 0)
    ```
- **Aggregation**:
  Because each customer has made different transactions, we need to aggregate to gather collective informations,
  'sum', 'mean','count', 'nunique', 'max', 'min' are the commonly used aggregation methods
  
- **Model training**:
  - StratifiedKold, split by target (to ensure each fold gets an equal proportion of target outliers)
  - lightgbm training, `rmse` metrics
  - Use optuna to optimize lightgbm parameters
  - Stack several lightgbm models to get final model
  
  
  
## What we could have done better:
  - Binary-classification of outliers, [Reference](https://www.kaggle.com/c/elo-merchant-category-recommendation/discussion/82036)





