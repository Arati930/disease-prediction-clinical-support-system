import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# ------------------ Page Config ------------------ #
st.set_page_config(page_title="AI Health Assistant", layout="wide")

# ------------------ Load Dataset ------------------ #
data = pd.read_csv("disease.csv")

# ------------------ Data Cleaning ------------------ #
data = data.drop_duplicates()
data = data.dropna(axis=1, how='all')
data = data.fillna(0)

# Convert all symptom columns to 0/1
for col in data.columns:
    if col != "prognosis":
        data[col] = data[col].apply(lambda x: 1 if x > 0 else 0)

# ------------------ Safe Disease Filtering ------------------ #
disease_counts = data["prognosis"].value_counts()

# Dynamic threshold (safe)
threshold = max(3, int(0.01 * len(data)))

common_diseases = disease_counts[disease_counts > threshold].index

if len(common_diseases) > 0:
    data = data[data["prognosis"].isin(common_diseases)]
else:
    st.warning("⚠️ Dataset too small, skipping filtering")

# ------------------ Safety Check ------------------ #
if data.shape[0] == 0:
    st.error("❌ Dataset is empty after cleaning. Check your dataset.")
    st.stop()

# ------------------ Features ------------------ #
X = data.drop("prognosis", axis=1)
y = data["prognosis"]

# Feature selection
temp_model = RandomForestClassifier()
temp_model.fit(X, y)

importances = pd.Series(temp_model.feature_importances_, index=X.columns)
top_features = importances.nlargest(15).index

X = X[top_features]

# ------------------ Train Model ------------------ #
model = RandomForestClassifier(n_estimators=150)
model.fit(X, y)

# ------------------ NLP Function ------------------ #
def text_to_symptoms(user_text, feature_list):
    user_text = user_text.lower()
    return [1 if feature.replace("_", " ") in user_text else 0 for feature in feature_list]

# ------------------ Recommendation ------------------ #
def get_recommendation(disease, severity):
    recommendations = {
        "Fungal infection": ["Keep area dry", "Use antifungal cream", "Avoid sharing items"],
        "Allergy": ["Avoid allergens", "Take antihistamines", "Keep environment clean"],
        "GERD": ["Avoid spicy food", "Eat small meals", "Stay upright after eating"]
    }

    general = ["Stay hydrated", "Rest well", "Eat balanced diet"]

    advice = recommendations.get(disease, general)

    if severity >= 6:
        advice.append("⚠️ Seek medical attention immediately")

    return advice

# ------------------ UI ------------------ #
st.markdown("<h1 style='text-align:center; color:#00ADB5;'>🧠 Clinical Decision Support System</h1>", unsafe_allow_html=True)

st.info("⚠️ This is an AI-based prediction system, not a medical diagnosis.")

# Chat input
st.markdown("### 🤖 Chat with AI")
user_text = st.text_input("Describe symptoms (e.g., fever and headache)")

# Layout
col1, col2 = st.columns([1, 2])

# ------------------ Input ------------------ #
with col1:
    st.subheader("🩺 Select Symptoms")

    user_input = []
    for col in X.columns:
        val = st.toggle(col)
        user_input.append(1 if val else 0)

    predict_btn = st.button("🔍 Predict")

# ------------------ Output ------------------ #
with col2:
    st.subheader("📊 Prediction Results")

    if predict_btn:

        # Chat override
        if user_text:
            user_input = text_to_symptoms(user_text, X.columns)

        # Minimum symptoms check
        if sum(user_input) < 3:
            st.warning("⚠️ Please select at least 3 symptoms")
            st.stop()

        result = model.predict([user_input])[0]
        proba = model.predict_proba([user_input])[0]
        confidence = max(proba) * 100
        severity = sum(user_input)

        # Low confidence check
        if confidence < 40:
            st.warning("⚠️ Low confidence prediction. Add more symptoms.")
            st.stop()

        # Result
        st.success(f"🩺 Predicted Disease: {result}")
        st.write(f"📊 Confidence: {confidence:.2f}%")

        # Top 3 predictions
        st.subheader("🔍 Top 3 Possible Diseases")
        top3_idx = proba.argsort()[-3:][::-1]

        for i in top3_idx:
            st.progress(float(proba[i]))
            st.write(f"{model.classes_[i]} ({proba[i]*100:.2f}%)")

        # Severity
        st.subheader("⚠️ Severity Level")
        if severity < 3:
            st.success("Mild")
        elif severity < 6:
            st.warning("Moderate")
        else:
            st.error("High - Consult Doctor")

        # Recommendations
        st.subheader("💡 Recommendations")
        for tip in get_recommendation(result, severity):
            st.info(f"✔ {tip}")