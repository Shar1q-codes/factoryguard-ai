import pandas as pd
import numpy as np
import joblib

def add_rolling_features(df, sensors, windows=[5, 10, 20]):
    """
    Create rolling mean, std, and exponential moving average
    for each sensor across given window sizes.
    Always group by unit_nr first!
    """
    for col in sensors:
        for w in windows:
            df[f'{col}_roll_mean_{w}'] = df.groupby('unit_nr')[col].transform(
                lambda x: x.rolling(w, min_periods=1).mean()
            )
            df[f'{col}_roll_std_{w}'] = df.groupby('unit_nr')[col].transform(
                lambda x: x.rolling(w, min_periods=1).std().fillna(0)
            )
            df[f'{col}_ema_{w}'] = df.groupby('unit_nr')[col].transform(
                lambda x: x.ewm(span=w, adjust=False).mean()
            )
    return df

def add_lag_features(df, sensors):
    """
    Add lag features t-1 and t-2 for each sensor.
    Fill missing lag values with 0.
    """
    for col in sensors:
        df[f'{col}_lag1'] = df.groupby('unit_nr')[col].transform(
            lambda x: x.shift(1)
        ).fillna(0)
        df[f'{col}_lag2'] = df.groupby('unit_nr')[col].transform(
            lambda x: x.shift(2)
        ).fillna(0)
    return df

def load_and_engineer(filepath='data/train_FD001_with_RUL.csv'):
    """
    Main function - load data, add all features, save pkl.
    """
    # Load data
    df = pd.read_csv(filepath)
    
    print("Data loaded:", df.shape)
    
    # Define sensors
    sensors = [f's{i}' for i in range(1, 22)]
    
    # Check which sensor columns actually exist
    sensors = [s for s in sensors if s in df.columns]
    
    # Also check sensor_1 format
    if not sensors:
        sensors = [f'sensor_{i}' for i in range(1, 22)]
        sensors = [s for s in sensors if s in df.columns]
    
    print(f"Using {len(sensors)} sensors:", sensors[:5], "...")
    
    # Add features
    df = add_rolling_features(df, sensors)
    df = add_lag_features(df, sensors)
    
    print("Features created! Total columns:", df.shape[1])
    
    # Save
    joblib.dump(df, 'data/features_df.pkl')
    print("Saved to data/features_df.pkl")
    
    return df

if __name__ == '__main__':
    df = load_and_engineer()
    print(df.shape)
    print(df.head(2))
