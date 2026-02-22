# ==========================================
# X-IDS Data Preprocessing & Model Training
# ==========================================

# Basic libraries
import pandas as pd
import numpy as np

# For preprocessing
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# Column names for NSL-KDD dataset
col_names = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes",
    "land","wrong_fragment","urgent","hot","num_failed_logins","logged_in",
    "num_compromised","root_shell","su_attempted","num_root","num_file_creations",
    "num_shells","num_access_files","num_outbound_cmds","is_host_login","is_guest_login",
    "count","srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate",
    "same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count",
    "dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate",
    "dst_host_serror_rate","dst_host_srv_serror_rate","dst_host_rerror_rate",
    "dst_host_srv_rerror_rate","label","difficulty"
]

# Load datasets
train_df = pd.read_csv(
    "C:\\Users\\Dell\\OneDrive\\Desktop\\Projects\\X-IDS\\nsl-kdd\\KDDTrain+.txt",
    names=col_names
)
test_df = pd.read_csv(
    "C:\\Users\\Dell\\OneDrive\\Desktop\\Projects\\X-IDS\\nsl-kdd\\KDDTest+.txt",
    names=col_names
)

# Drop 'difficulty' column if present
train_df = train_df.drop(columns=['difficulty'], errors='ignore')
test_df = test_df.drop(columns=['difficulty'], errors='ignore')

# Encode categorical columns
categorical_cols = ['protocol_type', 'service', 'flag']
for col in categorical_cols:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col])
    test_df[col] = le.transform(test_df[col])

# Map attack types to main categories
attack_map = {
    'normal': 'normal',

    # DoS
    'neptune':'dos','smurf':'dos','back':'dos','teardrop':'dos','land':'dos','pod':'dos',
    'apache2':'dos','udpstorm':'dos','processtable':'dos','worm':'dos',

    # Probe
    'satan':'probe','ipsweep':'probe','nmap':'probe','portsweep':'probe',
    'mscan':'probe','saint':'probe',

    # R2L
    'guess_passwd':'r2l','ftp_write':'r2l','imap':'r2l','phf':'r2l','multihop':'r2l',
    'warezmaster':'r2l','warezclient':'r2l','spy':'r2l','xlock':'r2l','xsnoop':'r2l',
    'snmpguess':'r2l','snmpgetattack':'r2l','httptunnel':'r2l',
    'sendmail':'r2l','named':'r2l',

    # U2R
    'buffer_overflow':'u2r','loadmodule':'u2r','rootkit':'u2r','perl':'u2r',
    'sqlattack':'u2r','xterm':'u2r','ps':'u2r'
}

# Apply attack mapping
train_df['attack_cat'] = train_df['label'].map(attack_map)
test_df['attack_cat'] = test_df['label'].map(attack_map)

# Drop rows with missing attack categories
train_df = train_df.dropna(subset=['attack_cat'])
test_df = test_df.dropna(subset=['attack_cat'])

# Separate features and labels
X_train = train_df.drop(['label', 'attack_cat'], axis=1)
X_test = test_df.drop(['label', 'attack_cat'], axis=1)
y_train = train_df['attack_cat']
y_test = test_df['attack_cat']

# Scale numeric features
scaler = MinMaxScaler()
X_train[X_train.columns] = scaler.fit_transform(X_train)
X_test[X_test.columns] = scaler.transform(X_test)

# Encode target labels
label_encoder = LabelEncoder()
label_encoder.fit(list(y_train) + list(y_test))
y_train_encoded = label_encoder.transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# ==========================================
# Train Random Forest & XGBoost
# ==========================================
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

# Train models
rf_model.fit(X_train, y_train_encoded)
xgb_model.fit(X_train, y_train_encoded)

# Evaluate
rf_preds = rf_model.predict(X_test)
xgb_preds = xgb_model.predict(X_test)

print("\n🔍 Random Forest Performance:\n", classification_report(y_test_encoded, rf_preds, target_names=label_encoder.classes_))
print("🔍 XGBoost Performance:\n", classification_report(y_test_encoded, xgb_preds, target_names=label_encoder.classes_))

# ==========================================
# Balance Dataset (SMOTE + Undersampling)
# ==========================================
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

smote = SMOTE(random_state=42)
undersample = RandomUnderSampler(random_state=42)

pipeline = Pipeline([
    ('smote', smote),
    ('undersample', undersample)
])

X_train_balanced, y_train_balanced = pipeline.fit_resample(X_train, y_train_encoded)

# Retrain on balanced data
rf_model.fit(X_train_balanced, y_train_balanced)
xgb_model.fit(X_train_balanced, y_train_balanced)

rf_preds = rf_model.predict(X_test)
xgb_preds = xgb_model.predict(X_test)

print("\n Final Evaluation After Balancing:")
print(" Random Forest Performance:\n", classification_report(y_test_encoded, rf_preds, target_names=label_encoder.classes_))
print(" XGBoost Performance:\n", classification_report(y_test_encoded, xgb_preds, target_names=label_encoder.classes_))


# ==========================================
# SHAP Explainability for IDLE
# ==========================================
import shap
import matplotlib.pyplot as plt

print("\nGenerating SHAP explainability visuals...")

# Create SHAP explainer for Random Forest
explainer_rf = shap.TreeExplainer(rf_model)
shap_values_rf = explainer_rf.shap_values(X_test)

# --- Summary Plot (Feature Importance)
plt.figure()
shap.summary_plot(shap_values_rf, X_test, plot_type="bar", show=False)
plt.title("Feature Importance (Random Forest)")
plt.tight_layout()
plt.show()  #  Forces plot to appear in IDLE

# --- Force Plot for One Sample
sample_index = 10
force_plot = shap.force_plot(
    explainer_rf.expected_value[0],
    shap_values_rf[0][sample_index],
    X_test.iloc[sample_index],
)

# Save the force plot as HTML (since IDLE can't render it)
shap.save_html("shap_force_plot.html", force_plot)
print("Saved SHAP interactive plot as shap_force_plot.html.")
print(" Open this file in your browser to view the explanation visually.")

