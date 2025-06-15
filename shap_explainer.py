import shap
import joblib
import pandas as pd

# Load pipeline and extract components
pipeline = joblib.load("fraud_job_model.pkl")
vectorizer = pipeline.named_steps['tfidf']
model = pipeline.named_steps['clf']

# Load and process data
df = pd.read_csv("NqndMEyZakuimmFI.csv")
df.fillna('', inplace=True)
df['text'] = df['title'] + ' ' + df['company_profile'] + ' ' + df['description'] + ' ' + df['requirements']

# Transform using TF-IDF and convert to dense
X_transformed = vectorizer.transform(df['text'].iloc[:50]).toarray()

# Use TreeExplainer (for RandomForest) with fix
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_transformed, check_additivity=False)

# Plot SHAP summary for fraudulent class (1)
shap.summary_plot(shap_values[1], feature_names=vectorizer.get_feature_names_out())
