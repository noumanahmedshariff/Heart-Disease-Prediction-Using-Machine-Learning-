import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ---------------- LOAD DATA ----------------
df = pd.read_csv("data/heart.csv")

X = df.drop("target", axis=1)

# ---------------- FIT ONE SCALER (IMPORTANT FIX) ----------------
# Fit scaler ONCE on full feature set
scaler = StandardScaler()
scaler.fit(X)

# ---------------- BINARY TARGET ----------------
# 0 = No disease, 1 = Disease
y_binary = df["target"].apply(lambda x: 1 if x > 0 else 0)

# ---------------- BINARY MODEL ----------------
X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
    X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
)

X_train_b_scaled = scaler.transform(X_train_b)
X_test_b_scaled = scaler.transform(X_test_b)

binary_model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42
)

binary_model.fit(X_train_b_scaled, y_train_b)

binary_pred = binary_model.predict(X_test_b_scaled)
print("Binary Accuracy:", accuracy_score(y_test_b, binary_pred))

# ---------------- SEVERITY MODEL (ONLY DISEASE CASES) ----------------
severity_df = df[df["target"] > 0]

X_sev = severity_df.drop("target", axis=1)
y_sev = severity_df["target"]  # 1, 2, 3, 4

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X_sev, y_sev, test_size=0.2, random_state=42, stratify=y_sev
)

X_train_s_scaled = scaler.transform(X_train_s)
X_test_s_scaled = scaler.transform(X_test_s)

severity_model = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=42
)

severity_model.fit(X_train_s_scaled, y_train_s)

sev_pred = severity_model.predict(X_test_s_scaled)
print("Severity Accuracy:", accuracy_score(y_test_s, sev_pred))

# ---------------- SAVE MODELS ----------------
joblib.dump(binary_model, "data/model_binary.pkl")
joblib.dump(severity_model, "data/model_severity.pkl")
joblib.dump(scaler, "data/scaler.pkl")

print("✅ Models trained and saved successfully")
