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

- **Local or Hosted Deployment**  
  Runs locally with Flask or can be hosted on any WSGI-compatible platform. 

---

## How It Works

1. **Chat & Emotion Detection**  
   - The bot greets the user and engages in three conversational turns.  
   - All three messages are sent to GPT-3.5 to classify mood into one of six emotions.

2. **Playlist Input**  
   - After a natural closing message, the bot asks for your Spotify playlist URL.  
   - It fetches track metadata (name, artist, Spotify ID) and audio features via the Spotify API.

3. **Recommendation**  
   - Matches your playlist tracks to a master dataset.  
   - Encodes both seed tracks and candidate tracks (filtered by detected emotion) via an autoencoder.  
   - Computes cosine similarity between the average seed embedding and all candidate embeddings.  
   - Returns the top 10 recommendations (excluding seeds), each with a direct Spotify URL.

---

## Deployment

### Hosted
You can deploy this Flask app on any hosting service (Render, Heroku, AWS, etc.). Be sure to set your environment variables:

```bash
SPOTIFY_CLIENT_ID=your_client_id
SPOTIFY_CLIENT_SECRET=your_client_secret
HF_TOKEN=your_hugging_face_key
OPENAI_API_KEY=your_openai_key
