# Music Recommendation Chatbot

This project is a **music recommendation chatbot** that provides personalized song suggestions based on user conversation and emotional context. It uses DistilRoBERTa-base for emotion analysis, fetches your Spotify playlist, and applies a pretrained autoencoder + cosine‐similarity filtering over audio features to generate tailored recommendations.

---

## Features

- **Emotion-Based Recommendations**  
  Analyzes three rounds of user chat with DistilRoBERTa-base to detect one of seven emotions (`anger`, `disgust`, `fear`, `neutral`, `joy`, `sadness`, `surprise`) and steers recommendations accordingly.

- **Playlist-Seeded Similarity Filtering**  
  After emotion detection, the user supplies a Spotify playlist URL. We scrape your playlist’s tracks, match them against our master dataset, encode them with an autoencoder, then recommend songs with high cosine similarity (excluding your seed tracks).

- **Lightweight, End-to-End Pipeline**  
  - DistilRoBERTa-base for quick emotion inference  
  - Spotipy for Spotify API integration  
  - PyTorch autoencoder + scikit-learn scaler for feature encoding  
  - Cosine similarity for nearest-neighbor retrieval
  - GPT-3.5 for user interactions

- **Local or Hosted Deployment**  
  Runs locally with Flask or can be hosted on any WSGI-compatible platform. 

---

## How It Works

1. **Chat & Emotion Detection**  
   - The bot greets the user and engages in three conversational turns.  
   - All three messages are sent to DistilRoBERTa-base to classify mood into one of seven emotions.

2. **Playlist Input**  
   - After a natural closing message, the bot asks for your Spotify playlist URL.  
   - It fetches track metadata (name, artist, Spotify ID) and audio features via the Spotify API.

3. **Recommendation**  
   - Matches your playlist tracks to a master dataset.  
   - Encodes both seed tracks and candidate tracks (filtered by detected emotion) via an autoencoder.  
   - Computes cosine similarity between the average seed embedding and all candidate embeddings.  
   - Returns the top 10 recommendations, each with a direct Spotify URL.

---

## Deployment & Tutorial

You can deploy this Flask app on any hosting service (e.g., Render, Heroku, AWS) or run it locally. Follow the steps below:

1.  Set the following environment variables (either via terminal or in a `.env` file):

    ```bash
    export SPOTIFY_CLIENT_ID=your_client_id
    export SPOTIFY_CLIENT_SECRET=your_client_secret
    export HF_API_KEY=your_hugging_face_key
    export OPENAI_API_KEY=your_openai_key
    ```
    If you choose to use a `.env` file, its content would look like this:
    ```env
    SPOTIFY_CLIENT_ID=your_client_id
    SPOTIFY_CLIENT_SECRET=your_client_secret
    HF_API_KEY=your_hugging_face_key
    OPENAI_API_KEY=your_openai_key
    ```

2. To download the dataset required for training and music recommendation, run:

    ```bash
    pip install gdown
    python download_data.py
    ```

3.  Run the Flask app:
    ```bash
    python app.py
    ```

4.  Open your browser and go to the local URL (e.g., `http://127.0.0.1:5000`) shown in your terminal.
