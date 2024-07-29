# Next-Word-Prediction-LSTM-and-GRU

Project Overview:
This project aims to develop a deep learning model for predicting the next word in a given sequence of words. The models are built using Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU) networks, which are well-suited for sequence prediction tasks.

Steps Involved:
1. Data Collection
2. Data Preprocessing
3. Model Building
4. Model Training
5. Model Evaluation
6. Deployment

1. Data Collection:
We use the text of Shakespeare's "Hamlet" as our dataset. This rich, complex text provides a good challenge for our model.

2. Data Preprocessing:
The text data is tokenized, converted into sequences, and padded to ensure uniform input lengths. The sequences are then split into training and testing sets.
Steps:
Tokenization: Converting text into tokens (words).
Sequences: Converting tokens into sequences.
Padding: Ensuring all sequences are of uniform length.

4. Model Building:
Two models are constructed: one using LSTM and the other using GRU. Each model consists of the following layers:
Embedding Layer: Converts input tokens into dense vectors of fixed size.
Recurrent Layers: Two LSTM/GRU layers for capturing temporal dependencies.
Dense Layer: Output layer with a softmax activation function to predict the probability of the next word.

4. Model Training:
The models are trained using the prepared sequences, with early stopping implemented to prevent overfitting. Early stopping monitors the validation loss and stops training when the loss stops improving.
Hyperparameters:
Loss Function: Categorical Cross-Entropy
Optimizer: Adam
Metrics: Accuracy

6. Model Evaluation:
The models are evaluated using a set of example sentences to test their ability to predict the next word accurately.

7. Deployment
A Streamlit web application is developed to allow users to input a sequence of words and get the predicted next word in real-time.

Installation
Clone the repository:
git clone https://github.com/bgomathidevi/next-word-prediction.git
cd next-word-prediction

Install the required packages:
pip install -r requirements.txt

Usage:
Train the models (optional if you use pre-trained models):
python train_model.py

Run the Streamlit app:
streamlit run app.py

Files:
train_model.py: Script to preprocess data and train the LSTM and GRU models.
app.py: Streamlit app for next word prediction.
tokenizer.pickle: Saved tokenizer for text preprocessing.
next_word_lstm.h5: Trained LSTM model.
next_word_gru.h5: Trained GRU model.
requirements.txt: Required Python packages.

Streamlit Application:
The Streamlit web application allows users to input a sequence of words and get the predicted next word in real-time. The interface includes:
Title: "Next Word Prediction Using LSTM and GRU"
Input Field: For entering the sequence of words.
Predict Button: To get the next word prediction.

Contributing:
Fork the repository.
Create your feature branch (git checkout -b feature/new-feature).
Commit your changes (git commit -m 'Add new feature').
Push to the branch (git push origin feature/new-feature).
Open a pull request.
