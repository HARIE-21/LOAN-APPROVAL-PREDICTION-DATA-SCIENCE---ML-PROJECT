# train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier
import joblib

print("Loading data...")
df = pd.read_csv("loan_dataset_20000.csv")

# Separate features and drop the 'grade_subgrade' shortcut!
X = df.drop(['loan_paid_back', 'grade_subgrade'], axis=1)
y = df['loan_paid_back']

categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', RobustScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_cols)
    ])

# Build and train the pipeline
print("Training XGBoost model... (This takes a few seconds)")
pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))
])

pipeline.fit(X, y)

# Save the model
joblib.dump(pipeline, 'xgboost_loan_model.pkl')
print("✅ SUCCESS: 'xgboost_loan_model.pkl' has been created in this folder!")