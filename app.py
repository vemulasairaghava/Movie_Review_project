import streamlit as st
import pickle
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="Movie Review Sentiment Analysis",
    page_icon="🎬",
    layout="wide"
)

# ---------------------------------------------------
# LOAD TOKENIZER
# ---------------------------------------------------

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# ---------------------------------------------------
# LOAD MODELS
# ---------------------------------------------------

simple_rnn = load_model("simple_rnn_model.keras")
lstm_model = load_model("lstm_model.keras")
gru_model = load_model("gru_model.keras")

MAX_LEN = 200

# ---------------------------------------------------
# PREPROCESS FUNCTION
# ---------------------------------------------------

def preprocess_text(text):

    sequence = tokenizer.texts_to_sequences([text])

    padded = pad_sequences(
        sequence,
        maxlen=MAX_LEN,
        padding='post',
        truncating='post'
    )

    return padded

# ---------------------------------------------------
# PREDICTION FUNCTION
# ---------------------------------------------------

def predict_review(review, model):

    processed = preprocess_text(review)

    probability = model.predict(
        processed,
        verbose=0
    )[0][0]

    sentiment = (
        "Positive"
        if probability >= 0.5
        else "Negative"
    )

    confidence = (
        probability
        if probability >= 0.5
        else 1 - probability
    )

    return sentiment, confidence, probability

# ---------------------------------------------------
# HEADER
# ---------------------------------------------------

st.title("🎬 Movie Review Sentiment Analysis System")

st.subheader(
    "Deep Learning Based Sentiment Classification"
)

st.markdown("---")

# ---------------------------------------------------
# MODEL SELECTION
# ---------------------------------------------------

selected_model = st.radio(
    "Select Model",
    ["SimpleRNN", "LSTM", "GRU"]
)

# ---------------------------------------------------
# INPUT AREA
# ---------------------------------------------------

review = st.text_area(
    "Enter your movie review here...",
    height=200
)

# ---------------------------------------------------
# ANALYZE BUTTON
# ---------------------------------------------------

if st.button("Analyze Review"):

    if len(review.strip()) == 0:

        st.warning(
            "Please enter a movie review."
        )

    else:

        if selected_model == "SimpleRNN":
            model = simple_rnn

        elif selected_model == "LSTM":
            model = lstm_model

        else:
            model = gru_model

        sentiment, confidence, prob = predict_review(
            review,
            model
        )

        st.markdown("## Prediction Result")

        st.success(
            f"Sentiment: {sentiment}"
        )

        st.info(
            f"Confidence: {confidence*100:.2f}%"
        )

        positive_prob = prob * 100
        negative_prob = (1 - prob) * 100

        st.markdown("## Probability Distribution")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Positive Probability",
                f"{positive_prob:.2f}%"
            )

        with col2:
            st.metric(
                "Negative Probability",
                f"{negative_prob:.2f}%"
            )

        st.markdown("## Confidence Chart")

        chart_df = pd.DataFrame({
            "Probability":[
                positive_prob,
                negative_prob
            ]
        },
        index=[
            "Positive",
            "Negative"
        ])

        st.bar_chart(chart_df)

# ---------------------------------------------------
# COMPARE ALL MODELS
# ---------------------------------------------------

st.markdown("---")

st.header("Compare Predictions from All Models")

if st.button("Compare All Models"):

    if len(review.strip()) == 0:

        st.warning(
            "Please enter a movie review."
        )

    else:

        results = []

        for model_name, model in [

            ("SimpleRNN", simple_rnn),

            ("LSTM", lstm_model),

            ("GRU", gru_model)

        ]:

            sentiment, confidence, _ = predict_review(
                review,
                model
            )

            results.append({
                "Model": model_name,
                "Sentiment": sentiment,
                "Confidence (%)":
                round(confidence * 100, 2)
            })

        result_df = pd.DataFrame(results)

        st.subheader("Model Comparison")

        st.dataframe(
            result_df,
            use_container_width=True
        )

        st.subheader(
            "Confidence Comparison"
        )

        confidence_chart = result_df.set_index(
            "Model"
        )[["Confidence (%)"]]

        st.bar_chart(confidence_chart)

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------

st.markdown("---")

st.caption(
    "Built using TensorFlow, Streamlit, SimpleRNN, LSTM and GRU"
)