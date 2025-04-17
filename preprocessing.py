import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_data(data):
    df = data.copy()
    df.drop(columns=["Source Name", "Destination Name", "Departure Date & Time", "Arrival Date & Time", "Total Fare (BDT)"], inplace=True, errors="ignore")

    df = df.infer_objects(copy=False)

    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str)
        df[col] = LabelEncoder().fit_transform(df[col])

    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = df[col].astype("float64")

    df.fillna(0, inplace=True)

    scaler = StandardScaler()
    temp = scaler.fit_transform(df)
    df = pd.DataFrame(temp, columns=df.columns)

    return df