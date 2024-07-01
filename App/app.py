# Core Pkgs
import streamlit as st
import altair as alt

# NLP pkgs
import numpy as np
import pandas as pd

# Utils
import joblib

# Update the path to the model file
model_path = 'models/emotion_pipe_lr.pkl'

try:
    pipe_lr = joblib.load(open(model_path, 'rb'))
except FileNotFoundError:
    st.error(f"Model file not found. Please ensure the file '{model_path}' exists.")

emotions_emoji_dict = {"anger":"😠","disgust":"🤮", "fear":"😨😱", "happy":"🤗", "joy":"😂", "neutral":"😐", "sad":"😔", "sadness":"😔", "shame":"😳", "surprise":"😮"}

def predict_emotion(docx):
    results = pipe_lr.predict([docx])
    return results[0]

def prediction_probability(docx):
    results = pipe_lr.predict_proba([docx])
    return results

def main():
    menu = ["Home", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.subheader('Emotion Text Classifier')
        with st.form(key="emotion_clf_form"):
            raw_text = st.text_area("Type Here")
            submit_text = st.form_submit_button(label="Submit")
        if submit_text:
            col1, col2 = st.columns(2)

            # Applying functions here
            prediction = predict_emotion(raw_text)
            probability = prediction_probability(raw_text)

            with col1:
                st.success("Original")
                st.write(raw_text)
                st.success("Prediction")
                emoji_icon = emotions_emoji_dict[prediction]
                st.write("{}:{}".format(prediction, emoji_icon))
                st.write("Confidence:{}".format(np.max(probability)))

            with col2:
                st.success("Prediction Probability")
                proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
                proba_df_clean = proba_df.T.reset_index()
                proba_df_clean.columns = ["Emotions", "Probability"]

                fig = alt.Chart(proba_df_clean).mark_bar().encode(x='Emotions', y='Probability', color='Emotions')
                st.altair_chart(fig, use_container_width=True)

    else:
        st.subheader("About")
        st.title("We are here to show How This Emotion Classifier Works..!!")

if __name__ == "__main__":
    main()
