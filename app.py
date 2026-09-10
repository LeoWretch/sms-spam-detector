import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="SMS Spam Detection",
    page_icon="📩",
    layout="centered"
)

# --------------------------------------------------
# NLP setup
# --------------------------------------------------

ps = PorterStemmer()


def transform_text(text):
    text = text.lower()

    text = nltk.word_tokenize(text)

    y = []

    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


# --------------------------------------------------
# Load model and vectorizer
# --------------------------------------------------

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))


# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("📩 SMS Spam Detection")

st.subheader("Detect spam messages using Machine Learning")

st.write(
    "Enter an SMS message below and our trained model will "
    "classify it as **Spam** or **Not Spam**."
)

st.divider()


# --------------------------------------------------
# Message input
# --------------------------------------------------

st.subheader("💬 Enter Your Message")

input_sms = st.text_area(
    "Message",
    placeholder="Example: Congratulations! You have won a free prize...",
    height=150,
    label_visibility="collapsed"
)

st.write("")


# --------------------------------------------------
# Predict button
# --------------------------------------------------

if st.button("🔍 Check Message", use_container_width=True):

    if input_sms.strip() == "":
        st.warning("⚠️ Please enter a message first.")

    else:

        with st.spinner("Analyzing message..."):

            # 1. Preprocess
            transform_sms = transform_text(input_sms)

            # 2. Vectorize
            vector_input = tfidf.transform([transform_sms])

            # 3. Predict
            result = model.predict(vector_input)[0]


        st.divider()

        # --------------------------------------------------
        # Result
        # --------------------------------------------------

        st.subheader("📊 Result")

        if result == 1:

            st.error("🚨 SPAM DETECTED")

            st.write(
                "This message has been classified as **Spam** "
                "by the machine learning model."
            )

        else:

            st.success("✅ NOT SPAM")

            st.write(
                "This message has been classified as **Not Spam** "
                "by the machine learning model."
            )


# --------------------------------------------------
# Information section
# --------------------------------------------------

st.divider()

st.subheader("🤖 About This Project")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Technology", "Python")

with col2:
    st.metric("Feature Extraction", "TF-IDF")

with col3:
    st.metric("Task", "Classification")


# --------------------------------------------------
# How it works
# --------------------------------------------------

with st.expander("🔎 How does it work?"):

    st.write("""
    **1. Text Preprocessing**

    The message is converted to lowercase, tokenized,
    stopwords are removed, and stemming is performed.

    **2. TF-IDF**

    The processed text is converted into numerical features
    using TF-IDF.

    **3. Machine Learning**

    The trained classification model predicts whether the
    message is Spam or Not Spam and Model Precision is 100% and tested.

    **4. Result**

    The prediction is displayed above.
    """)


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.caption(
    "📩 SMS Spam Detection • NLP • TF-IDF • Machine Learning"
)