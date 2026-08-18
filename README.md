# 🎬 IMDB Movie Reviews Sentiment Analysis Using LSTM

A deep learning project that uses a **Long Short-Term Memory (LSTM)** neural network to classify IMDB movie reviews as either **Positive** or **Negative**.

This project demonstrates an end-to-end Natural Language Processing workflow, including dataset loading, sentiment encoding, text tokenization, sequence conversion, padding, LSTM model development, training, evaluation, and prediction on new movie reviews.

---

## 📌 Overview

The project performs the following tasks:

- Downloads the IMDB movie review dataset using the Kaggle API.
- Loads and explores the dataset.
- Converts sentiment labels into numerical values.
- Splits the dataset into training and testing sets.
- Tokenizes review text.
- Converts text into numerical sequences.
- Pads sequences to a fixed length.
- Builds an LSTM-based neural network.
- Trains the model using the prepared review data.
- Evaluates the model on unseen test data.
- Provides a function for predicting the sentiment of new reviews.

---

## 🎯 Objective

The objective of this project is to build a deep learning model that can understand textual patterns in movie reviews and classify each review as either positive or negative.

### Sentiment Labels

| Label | Sentiment |
|---:|---|
| `0` | Negative |
| `1` | Positive |

---

## 🧠 Technologies Used

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Deep Learning Framework | TensorFlow, Keras |
| Neural Network | LSTM |
| Text Processing | Keras Tokenizer, Sequence Padding |
| Dataset Access | Kaggle API |
| Development Environment | Jupyter Notebook, Google Colab |

---

## 📂 Dataset

The project uses the **IMDB Dataset of 50K Movie Reviews**, obtained through the Kaggle API.

The dataset contains movie reviews divided into two sentiment categories:

```text
positive
negative
```

### Dataset Details

| Property | Details |
|---|---|
| Dataset Name | IMDB Dataset of 50K Movie Reviews |
| Total Reviews | 50,000 |
| Number of Classes | 2 |
| Review Type | Text |
| Positive Label | `1` |
| Negative Label | `0` |
| Dataset Source | Kaggle |

### Dataset Source

Add the exact dataset URL to this section if you publish the repository.

---

## 🔄 Project Workflow

```text
IMDB Movie Reviews
        │
        ▼
Data Collection
        │
        ▼
Dataset Loading
        │
        ▼
Sentiment Label Encoding
        │
        ▼
Train-Test Split
        │
        ▼
Text Tokenization
        │
        ▼
Sequence Conversion
        │
        ▼
Sequence Padding
        │
        ▼
Embedding Layer
        │
        ▼
LSTM Layer
        │
        ▼
Sigmoid Output
        │
        ▼
Model Training
        │
        ▼
Model Evaluation
        │
        ▼
Sentiment Prediction
```

---

## 📊 Data Preparation

The dataset is loaded into a Pandas DataFrame.

Sentiment labels are converted from text into numerical values:

```python
data.replace(
    {
        "sentiment": {
            "positive": 1,
            "negative": 0
        }
    },
    inplace=True
)
```

The dataset is then divided into training and testing sets using an 80:20 split:

```python
from sklearn.model_selection import train_test_split

train_data, test_data = train_test_split(
    data,
    test_size=0.2,
    random_state=42
)
```

---

## 🔤 Text Preprocessing

The review text is converted into numerical sequences before being passed to the LSTM network.

### Tokenization

A Keras `Tokenizer` is used with a vocabulary size of 5,000 words:

```python
from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(num_words=5000)

tokenizer.fit_on_texts(
    train_data["review"]
)
```

### Sequence Conversion

The text reviews are converted into integer sequences:

```python
X_train = tokenizer.texts_to_sequences(
    train_data["review"]
)

X_test = tokenizer.texts_to_sequences(
    test_data["review"]
)
```

### Sequence Padding

All sequences are padded or truncated to a maximum length of 200 tokens:

```python
from tensorflow.keras.preprocessing.sequence import pad_sequences

X_train = pad_sequences(
    X_train,
    maxlen=200
)

X_test = pad_sequences(
    X_test,
    maxlen=200
)
```

Padding ensures that every review has the same input size for the neural network.

---

## 🏗️ LSTM Model Architecture

The model is built using TensorFlow and Keras.

```text
Input Sequence
      │
      ▼
Embedding Layer
      │
      ▼
LSTM Layer
      │
      ▼
Dense Output Layer
      │
      ▼
Sigmoid Activation
      │
      ▼
Positive or Negative
```

### Model Configuration

| Layer | Configuration |
|---|---|
| Embedding | Input dimension: 5,000, output dimension: 128 |
| LSTM | 128 units |
| Dropout | 0.2 |
| Recurrent Dropout | 0.2 |
| Output Layer | Dense layer with sigmoid activation |

### Model Definition

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

model = Sequential()

model.add(
    Embedding(
        input_dim=5000,
        output_dim=128,
        input_length=200
    )
)

model.add(
    LSTM(
        128,
        dropout=0.2,
        recurrent_dropout=0.2
    )
)

model.add(
    Dense(
        1,
        activation="sigmoid"
    )
)
```

---

## ⚙️ Model Compilation

The model is compiled using the following configuration:

| Parameter | Configuration |
|---|---|
| Optimizer | Adam |
| Loss Function | Binary Crossentropy |
| Evaluation Metric | Accuracy |

```python
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
```

---

## 🚀 Model Training

The model is trained using the following configuration:

| Parameter | Value |
|---|---:|
| Training Epochs | 5 |
| Batch Size | 64 |
| Validation Split | 20% |

```python
history = model.fit(
    X_train,
    Y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.2
)
```

The validation split allows the model's performance to be monitored during training.

---

## 📈 Model Evaluation

After training, the model is evaluated using the test dataset:

```python
loss, accuracy = model.evaluate(
    X_test,
    Y_test
)

print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")
```

The notebook calculates the final test loss and test accuracy when it is executed.

> **Important:** This README intentionally does not claim a fixed accuracy because the notebook source does not provide a recorded final evaluation value. Add the actual accuracy only after running the notebook.

### Evaluation Metrics

- Test loss.
- Test accuracy.
- Training loss.
- Validation loss.
- Training accuracy.
- Validation accuracy.

---

## 🔍 Sentiment Prediction

A prediction function is created to classify new movie reviews.

### Prediction Pipeline

```text
New Review
    │
    ▼
Text Tokenization
    │
    ▼
Sequence Conversion
    │
    ▼
Sequence Padding
    │
    ▼
LSTM Model
    │
    ▼
Sigmoid Probability
    │
    ▼
Sentiment Classification
```

### Classification Threshold

The prediction threshold used by the model is:

```text
Prediction > 0.5 → Positive
Prediction ≤ 0.5 → Negative
```

### Example Prediction

```python
new_review = "This movie was fantastic. I loved it."

sentiment = predict_sentiment(
    new_review
)

print(
    f"The sentiment of the review is: {sentiment}"
)
```

### Example Outputs

```text
The sentiment of the review is: Positive
```

or:

```text
The sentiment of the review is: Negative
```

The notebook also demonstrates predictions using reviews containing negative or mixed wording.

---

## 🛠️ Prediction Function

A prediction function can be implemented as follows:

```python
def predict_sentiment(review):
    sequence = tokenizer.texts_to_sequences([review])

    padded_sequence = pad_sequences(
        sequence,
        maxlen=200
    )

    prediction = model.predict(
        padded_sequence
    )

    if prediction > 0.5:
        return "Positive"
    else:
        return "Negative"
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd <YOUR_REPOSITORY_NAME>
```

### 2. Install Dependencies

```bash
pip install kaggle pandas scikit-learn tensorflow
```

### 3. Configure Kaggle API

The notebook uses the Kaggle API to download the dataset.

Place your Kaggle API credentials file in the appropriate location and configure the credentials before running the dataset download step.

> ⚠️ **Security warning:** Never commit `kaggle.json`, API keys, passwords, or other credentials to GitHub.

### 4. Run the Notebook

Open the notebook using Jupyter:

```bash
jupyter notebook
```

Alternatively, upload the notebook to Google Colab.

Execute the notebook cells sequentially.

---

## 📁 Suggested Repository Structure

```text
IMDB-Sentiment-Analysis/
│
├── IMDB reviews - Sentiment Analysis - LSTM.ipynb
├── README.md
├── requirements.txt
│
└── data/
    └── IMDB Dataset.csv
```

The dataset can be downloaded through the Kaggle API instead of being committed to the repository.

---

## 📦 Requirements

A `requirements.txt` file can contain:

```text
kaggle
pandas
scikit-learn
tensorflow
numpy
```

Install the requirements using:

```bash
pip install -r requirements.txt
```

---

## ⚠️ Limitations

This project is primarily intended for educational purposes.

Potential limitations include:

- Model performance depends on the quality and distribution of the training data.
- The model may struggle with sarcasm.
- Irony and humor may be difficult to classify.
- Slang and informal language may reduce prediction reliability.
- Reviews containing mixed sentiments may be challenging.
- The model is limited to binary sentiment classification.
- Reviews containing words or expressions outside the training vocabulary may be less reliable.
- Predictions on text that differs significantly from the training data may have lower accuracy.
- The model does not provide detailed explanations for its predictions.

---

## 🔮 Future Improvements

Possible improvements include:

- Using a larger vocabulary.
- Performing hyperparameter tuning.
- Adding Bidirectional LSTM layers.
- Using pretrained word embeddings.
- Experimenting with GRU-based architectures.
- Applying attention mechanisms.
- Using Transformer-based models such as BERT.
- Adding precision, recall, and F1-score.
- Generating a confusion matrix.
- Adding training and validation accuracy plots.
- Adding training and validation loss plots.
- Implementing early stopping.
- Applying learning-rate scheduling.
- Handling class imbalance if present.
- Deploying the model as a web application.
- Creating a REST API for sentiment prediction.
- Building an interactive sentiment analysis interface.

---

## 📚 Learning Outcomes

This project provides practical experience with:

- Natural Language Processing.
- Text classification.
- Sentiment analysis.
- Tokenization.
- Sequence preprocessing.
- Sequence padding.
- Word embeddings.
- LSTM networks.
- Binary classification.
- Model training.
- Model evaluation.
- Validation monitoring.
- Sentiment prediction using deep learning.
- TensorFlow and Keras.
- Pandas and NumPy data processing.
- Scikit-learn dataset splitting.

---

## 👨‍💻 Author

```text
Your Name
```

- GitHub: [Your GitHub Profile](<YOUR_GITHUB_PROFILE>)
- LinkedIn: [Your LinkedIn Profile](<YOUR_LINKEDIN_PROFILE>)

---

## 📄 License

This project is intended for educational purposes.

If you plan to distribute or modify the project, add an appropriate open-source license after checking the licensing terms of the dataset and other resources used.

---

## ⭐ Acknowledgements

- [Kaggle](https://www.kaggle.com/)
- [TensorFlow](https://www.tensorflow.org/)
- [Keras](https://keras.io/)
- [Pandas](https://pandas.pydata.org/)
- [Scikit-learn](https://scikit-learn.org/)
- [NumPy](https://numpy.org/)
- IMDB dataset contributors

---

⭐ If you found this project useful, consider giving the repository a star!
