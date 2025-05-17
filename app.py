# app_new.py

import os
import json
import openai
from flask import Flask, request, jsonify, render_template
from recommender_new import fetch_playlist_df, recommend_songs

# Configure OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)

# 简易全局状态（Demo 用），生产环境请改为会话存储
chat_history = []
user_msgs = []
emotion = None
chat_count = 0


def detect_emotion(conversations):
    EMOTIONS = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
    system_prompt = (
        "You are an emotion classification assistant. "
        "Given the user input, classify the predominant emotion as one of: "
        + ", ".join(EMOTIONS)
        + "."
    )
    messages = [{"role": "system", "content": system_prompt}]
    for turn in conversations:
        messages.append({"role": "user", "content": turn})
    resp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=messages, max_tokens=5, temperature=0
    )
    emo = resp.choices[0].message.content.strip().lower()
    return emo if emo in EMOTIONS else "joy"


def chat_reply(history):
    resp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=history, max_tokens=150, temperature=0.7
    )
    return resp.choices[0].message.content.strip()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/get")
def get_message():
    global chat_count, emotion

    msg = request.args.get("msg", "").strip()

    # 如果是 Spotify 链接，直接返回推荐列表 JSON
    if msg.lower().startswith("http"):
        if emotion is None:
            emotion = "joy"
        # 爬取歌单并推荐
        df = fetch_playlist_df(msg)
        recs = recommend_songs(df, emotion)
        # 构建纯 JSON 数组
        output = []
        for _, row in recs.iterrows():
            output.append({
                "name": row["name"],
                "artist": row["artist"],
                "url": f"https://open.spotify.com/track/{row['id']}"
            })
        # 不再转成字符串，直接返回数组
        return jsonify(response=output)

    # 否则继续聊天逻辑
    chat_count += 1
    chat_history.append({"role": "user", "content": msg})

    # 收集前三条消息用于情绪检测
    if chat_count <= 3:
        user_msgs.append(msg)
        reply = chat_reply(chat_history)
        chat_history.append({"role": "assistant", "content": reply})
        # 第三条后，用一句收尾语提示用户输入 URL
        if chat_count == 3:
            return jsonify(response=reply + " Thanks! Now please send me your Spotify playlist URL.")
        else:
            return jsonify(response=reply)

    # 安全兜底：如果超出计数，重置对话
    return jsonify(response="Let's start over. How are you feeling today?")

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
