# IMDB Sentiment Analysis Web App
This repository contains a sentiment analysis web application built with TensorFlow and Streamlit. The app predicts the sentiment of movie reviews, classifying them as positive, negative, or neutral based on the text of the review. It uses a pre-trained LSTM model on the IMDB dataset.

## Features
- Sentiment prediction for movie reviews
- Classification into positive, negative, or neutral sentiments
- Simple and interactive user interface

## Installation
To run this app locally, you'll need to have Python installed on your system. Follow these steps to set up and start the app:
1. Clone the repository
```bash
   git clone https://github.com/YadamDheerajReddy/MovieSentimentAnalysisApp
   cd MovieSentimentAnalysisApp
```
2. Create and activate a virtual environment (optional but recommended)
   - For Unix or MacOS:
     ```bash
       python3 -m venv venv
       source venv/bin/activate
     ```
   - For Windows:
     ```bash
       python -m venv venv
       .\venv\Scripts\activate
     ```
3. Install the requirements
   ```bash
     pip install -r requirements.txt
   ```
4. Run the Streamlit app
   ```bash
     streamlit run sentiment_analysis_app.py
   ```

## How It Works
The app uses a pre-trained LSTM (Long Short-Term Memory) model to analyze and predict the sentiment of input movie reviews. It classifies sentiments into three categories: positive, negative, and neutral. The classification is based on the predicted probability output from the model, with a customizable threshold for neutral sentiments.

## Usage
After starting the app, navigate to the displayed URL (usually http://localhost:8501). Enter a movie review into the text area and click the "Predict Sentiment" button to see the sentiment analysis.

## Training the Model
The LSTM model was trained on the IMDB movie review dataset. See `imdb.py` for the script used to train and save the model. Adjustments to the model architecture or training parameters can be made in this script.

## Contributing
Contributions to improve the app or model are welcome. Please follow the standard fork-and-pull request workflow.
- Fork the repository
- Create your feature branch (`git checkout -b feature/AmazingFeature`)
- Commit your changes (`git commit -am 'Add some AmazingFeature'`)
- Push to the branch (`git push origin feature/AmazingFeature`)
- Open a pull request

## Acknowledgments
- TensorFlow and Keras for model building and training
- Streamlit for creating the web application
- The IMDB dataset provided by TensorFlow/Keras
