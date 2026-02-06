import pandas as pd


df = pd.read_csv(
    "sms.csv",
    sep=";",
    encoding="latin1",
    engine="python"
)

# Afisare 10 randuri random
random_rows = df.sample(n=10).reset_index(drop=True)

print(random_rows)

df = df.drop(columns=["id"])
df.columns
receiver_parts = df["receiver"].astype(str).str.split("-", expand=True)

df["country"] = receiver_parts[0]
df["operator"] = receiver_parts[1] + "-" + receiver_parts[2]
df["number"] = receiver_parts[3]
df["sent_date"] = pd.to_datetime(df["sent_date"], errors="coerce")

df["year"] = df["sent_date"].dt.year
df["month"] = df["sent_date"].dt.month
df["day"] = df["sent_date"].dt.day
df["hour"] = df["sent_date"].dt.hour
features_df = df[
    [
        "sender",
        "country",
        "operator",
        "number",
        "message",
        "year",
        "month",
        "day",
        "hour"
    ]
]

#features_df.head()
#doar asa

random_features = features_df.sample(n=10).reset_index(drop=True)

print(random_features)
features_df.info()
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import IsolationForest

df_ml = features_df.copy()

le_sender = LabelEncoder()
le_country = LabelEncoder()
le_operator = LabelEncoder()

df_ml["sender_enc"] = le_sender.fit_transform(df_ml["sender"])
df_ml["country_enc"] = le_country.fit_transform(df_ml["country"])
df_ml["operator_enc"] = le_operator.fit_transform(df_ml["operator"])

X = df_ml[
    [
        "sender_enc",
        "country_enc",
        "operator_enc",
        "year",
        "month",
        "day",
        "hour"
    ]
]

iso = IsolationForest(
    n_estimators=500,
    contamination=0.05,
    random_state=42
)

df_ml["anomaly"] = iso.fit_predict(X)
df_ml["anomaly_score"] = iso.decision_function(X)

#df_ml[["sender", "country", "operator", "hour", "anomaly", "anomaly_score"]].head(25)

df_ml[
    ["sender", "country", "operator", "hour", "anomaly", "anomaly_score"]
].sample(n=25).reset_index(drop=True)

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import IsolationForest
from scipy.stats import zscore

df_ml = features_df.copy()

# Eliminam coloana id
if "id" in df_ml.columns:
    df_ml = df_ml.drop(columns=["id"])


for col in ["sender", "country", "operator"]:
    df_ml[col + "_enc"] = LabelEncoder().fit_transform(df_ml[col])

# caracteristici
X = df_ml[["sender_enc", "country_enc", "operator_enc", "year", "month", "day", "hour"]]


iso = IsolationForest(n_estimators=500, contamination=0.05, random_state=42)
df_ml["anomaly"] = iso.fit_predict(X)  # -1 = anomalie, 1 = normal
df_ml["anomaly_score"] = iso.decision_function(X)

# Z-score
df_ml["z_score"] = zscore(df_ml["anomaly_score"])


df_ml["anomaly_flag"] = df_ml["z_score"] < -2  # True = anomalie


print(df_ml[["sender","country","operator","hour","anomaly","z_score","anomaly_flag"]].sample(10))


procent_anomalie = df_ml["anomaly_flag"].mean()
print(f"Procentul de anomalii detectate: {procent_anomalie:.2%}")
import pandas as pd
import numpy as np

X = df_ml[
    [
        "sender_enc",
        "country_enc",
        "operator_enc",
        "year",
        "month",
        "day",
        "hour"
    ]
]
from sklearn.ensemble import IsolationForest

iso = IsolationForest(
    n_estimators=200,
    contamination=0.05,   # 5% suspecte
    random_state=42
)

df_ml["if_label"] = iso.fit_predict(X)

# -1 = anomalie -> frauda
#  1 = normal
df_ml["fraud"] = (df_ml["if_label"] == -1).astype(int)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    df_ml["fraud"],
    test_size=0.2,
    random_state=42,
    stratify=df_ml["fraud"]
)
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=20,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)
df_ml["fraud_probability"] = rf.predict_proba(X)[:, 1]
feature_importance = pd.Series(
    rf.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\n=== Feature importance ===")
print(feature_importance)
suspect_sms = df_ml[df_ml["fraud_probability"] > 0.7]
result = suspect_sms[
    [
        "sender",
        "country",
        "operator",
        "year",
        "month",
        "day",
        "hour",
        "fraud_probability"
    ]
].sample(n=25).reset_index(drop=True)

#
result["fraud_probability"] = result["fraud_probability"].round(3)

print("\n=== 25 SMS suspecte (random forest) ===")
print(result.to_string(index=False))






import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split



# Features
X = df_ml[
    [
        "sender_enc",
        "country_enc",
        "operator_enc",
        "year",
        "month",
        "day",
        "hour"
    ]
]


iso = IsolationForest(
    n_estimators=200,
    contamination=0.05,   # 5% suspecte
    random_state=42
)

df_ml["if_label"] = iso.fit_predict(X)
# -1 = anomalie (posibil frauda)
#  1 = normal


df_ml["fraud"] = (df_ml["if_label"] == -1).astype(int)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    df_ml["fraud"],
    test_size=0.2,
    random_state=42,
    stratify=df_ml["fraud"] 
)


rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=20,
    class_weight="balanced",  
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)


df_ml["fraud_probability"] = rf.predict_proba(X)[:, 1]


feature_importance = pd.Series(
    rf.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\n=== Feature importance ===")
print(feature_importance)


#  SMS suspecte (probabilitate > 0.7)

suspect_sms = df_ml[df_ml["fraud_probability"] > 0.7]

result = suspect_sms[
    [
        "sender",
        "country",
        "operator",
        "year",
        "month",
        "day",
        "hour",
        "fraud_probability"
    ]
].sample(n=25).reset_index(drop=True)


result["fraud_probability"] = result["fraud_probability"].round(3)

print("\n=== 25 SMS suspecte (Random Forest) ===")
print(result.to_string(index=False))










import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from scipy.stats import zscore



# Features
X = df_ml[
    ["sender_enc", "country_enc", "operator_enc", "year", "month", "day", "hour"]
]


iso = IsolationForest(
    n_estimators=200,
    contamination=0.05,
    random_state=42
)

df_ml["if_label"] = iso.fit_predict(X)
df_ml["fraud"] = (df_ml["if_label"] == -1).astype(int)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    df_ml["fraud"],
    test_size=0.2,
    random_state=42,
    stratify=df_ml["fraud"]
)

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=20,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)


df_ml["fraud_probability"] = rf.predict_proba(X)[:, 1]


df_ml["z_score"] = zscore(df_ml["fraud_probability"])

# Marcare SMS suspecte folosind Z-score
df_ml["fraud_flag_z"] = df_ml["z_score"] > 2  

sample_df = df_ml[[
    "sender", "country", "operator", "hour", "fraud_probability", "z_score", "fraud_flag_z"
]].sample(25).reset_index(drop=True)

print("\n=== 25 SMS cu Z-score (Random Forest) ===")
print(sample_df)


procent_suspecte = df_ml["fraud_flag_z"].mean()
print(f"\nProcentul SMS suspecte (Z-score > 2): {procent_suspecte:.2%}")




X = df_ml[
    [
        "sender_enc",
        "country_enc",
        "operator_enc",

        "year",
        "month",
        "day",
        "hour"
    ]
]
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
from sklearn.neighbors import NearestNeighbors
import numpy as np

k = 15

nn = NearestNeighbors(
    n_neighbors=k,
    metric="euclidean"
)

nn.fit(X_scaled)

distances, _ = nn.kneighbors(X_scaled)

# scor de anomalie = distanta medie fata de vecini
df_ml["knn_anomaly_score"] = distances.mean(axis=1)
threshold = df_ml["knn_anomaly_score"].quantile(0.95)

df_ml["knn_fraud"] = (df_ml["knn_anomaly_score"] > threshold).astype(int)
df_ml["knn_risk"] = (
    df_ml["knn_anomaly_score"] - df_ml["knn_anomaly_score"].min()
) / (
    df_ml["knn_anomaly_score"].max() - df_ml["knn_anomaly_score"].min()
)
suspect_sms = df_ml[df_ml["knn_fraud"] == 1]

result = suspect_sms[
    [
        "sender",
        "country",
        "operator",

        "day",
        "hour",
        "knn_risk"
    ]
].sample(n=25).reset_index(drop=True)

result["knn_risk"] = result["knn_risk"].round(3)

print("\n=== 25 SMS suspecte (DOAR KNN) ===")
print(result.to_string(index=False))













X = df_ml[
    [
        "sender_enc",
        "country_enc",
        "operator_enc",
        "year",
        "month",
        "day",
        "hour"
    ]
]


#  SCALARE


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)



from sklearn.neighbors import NearestNeighbors
import numpy as np

k = 15

nn = NearestNeighbors(
    n_neighbors=k,
    metric="euclidean"
)

nn.fit(X_scaled)

distances, _ = nn.kneighbors(X_scaled)

# scor de anomalie = distanta medie fata de vecini
df_ml["knn_anomaly_score"] = distances.mean(axis=1)

# prag KNN (top 5% anomalii)
knn_threshold = df_ml["knn_anomaly_score"].quantile(0.95)
df_ml["knn_fraud"] = (df_ml["knn_anomaly_score"] > knn_threshold).astype(int)

# scor de risc KNN normalizat (0–1)
df_ml["knn_risk"] = (
    df_ml["knn_anomaly_score"] - df_ml["knn_anomaly_score"].min()
) / (
    df_ml["knn_anomaly_score"].max() - df_ml["knn_anomaly_score"].min()
)



from scipy.stats import zscore

df_ml["knn_zscore"] = zscore(df_ml["knn_anomaly_score"])

# prag Z-score
z_threshold = 3
df_ml["z_fraud"] = (df_ml["knn_zscore"] > z_threshold).astype(int)

# scor risc Z-score normalizat
df_ml["z_risk"] = (
    df_ml["knn_zscore"] - df_ml["knn_zscore"].min()
) / (
    df_ml["knn_zscore"].max() - df_ml["knn_zscore"].min()
)



df_ml["fraud_both"] = (
    (df_ml["knn_fraud"] == 1) &
    (df_ml["z_fraud"] == 1)
).astype(int)



# KNN 
suspect_knn = df_ml[df_ml["knn_fraud"] == 1]

result_knn = suspect_knn[
    [
        "sender",
        "country",
        "operator",
        "day",
        "hour",
        "knn_risk"
    ]
].sample(n=min(25, len(suspect_knn))).reset_index(drop=True)

result_knn["knn_risk"] = result_knn["knn_risk"].round(3)

print("\n=== 25 SMS suspecte (KNN) ===")
print(result_knn.to_string(index=False))

# --- Z-SCORE ---
suspect_z = df_ml[df_ml["z_fraud"] == 1]

result_z = suspect_z[
    [
        "sender",
        "country",
        "operator",
        "day",
        "hour",
       # "z_risk",
        "knn_zscore"
    ]
].sample(n=min(25, len(suspect_z))).reset_index(drop=True)

#result_z["z_risk"] = result_z["z_risk"].round(3)
result_z["knn_zscore"] = result_z["knn_zscore"].round(2)

print("\n=== 25 SMS suspecte (Z-SCORE) ===")
print(result_z.to_string(index=False))

# --- AMBELE ---
high_confidence = df_ml[df_ml["fraud_both"] == 1]

print("\n=== SMS suspecte CONFIRMATE (KNN + Z) ===")
print("Total:", len(high_confidence))

result_both = high_confidence[
    [
        "sender",
        "country",
        "operator",
        "day",
        "hour",
        "knn_risk"
       # "z_risk"
    ]
].sample(n=min(25, len(high_confidence))).reset_index(drop=True)

result_both["knn_risk"] = result_both["knn_risk"].round(3)
#result_both["z_risk"] = result_both["z_risk"].round(3)

print(result_both.to_string(index=False))
