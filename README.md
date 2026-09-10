# 📱 SMS Spam Detection

A machine learning based web application that detects whether an SMS message is **Spam** or **Not Spam** using Natural Language Processing (NLP) and the **Multinomial Naive Bayes** algorithm.

## 🚀 Features

- Detects spam and legitimate SMS messages
- Text preprocessing using NLTK
- TF-IDF based text vectorization
- Multinomial Naive Bayes classification
- Interactive Streamlit web interface
- Real-time prediction

## 🛠️ Technologies Used

- **Python**
- **Streamlit** – Web application
- **Scikit-learn** – Machine Learning
- **NLTK** – Natural Language Processing
- **TF-IDF** – Feature extraction
- **Multinomial Naive Bayes** – Classification algorithm
- **Pickle** – Model and vectorizer serialization

## 📂 Project Structure

```text
sms-spam-detector/
│
├── app.py
├── main.py
├── model.pkl
├── vectorizer.pkl
├── requirements.txt
├── README.md
└── .gitignore
```

## 🔄 How It Works

The application follows these steps:

1. User enters an SMS message.
2. The message is converted to lowercase.
3. Text is tokenized using NLTK.
4. Punctuation and non-alphanumeric tokens are removed.
5. English stopwords are removed.
6. Words are stemmed using Porter Stemmer.
7. The processed text is converted into numerical features using TF-IDF.
8. The Multinomial Naive Bayes model predicts whether the message is **Spam** or **Not Spam**.

## ▶️ Run Locally

### 1. Clone the Repository

```bash
git clone https://github.com/LeoWretch/sms-spam-detector.git
```

### 2. Navigate to the Project

```bash
cd sms-spam-detector
```

### 3. Create a Virtual Environment

```bash
python -m venv .venv
```

### 4. Activate the Virtual Environment

**Windows:**

```bash
.venv\Scripts\activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Streamlit Application

```bash
streamlit run app.py
```

The application will open in your browser.

## 🤖 Machine Learning Model

The project uses **Multinomial Naive Bayes** for SMS text classification.

### Model

The trained model is stored in:

```text
model.pkl
```

### Vectorizer

The TF-IDF vectorizer is stored in:

```text
vectorizer.pkl
```

## 📌 Example

### 🚨 Spam Message

**Input:**

```text
Congratulations! You have won a free lottery ticket. Call now!
```

**Prediction:**

🚨 **Spam Detected**

### ✅ Normal Message

**Input:**

```text
Hey, are we meeting tomorrow at 10 AM?
```

**Prediction:**

✅ **Not Spam**

## 🔮 Future Improvements

- Add prediction confidence/probability
- Support multiple languages
- Improve the user interface
- Deploy the application online
- Improve model performance with additional training data
- Add model performance visualizations

## 👨‍💻 Author

**LeoWretch**

⭐ If you find this project useful, consider giving it a star!
