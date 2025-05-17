document.addEventListener("DOMContentLoaded", () => {
    const chatWindow = document.getElementById("chat-window");
    const messagesContainer = document.getElementById("messages");
    const userInput = document.getElementById("userInput");
    const sendButton = document.getElementById("sendButton");
    const recommendationsContainer = document.getElementById("recommendations");
    const loadingContainer = document.getElementById("loading-container");

    // 1 = emotion chat, 2 = waiting for playlist URL, 3 = done
    let chatStage = 1;
    let chatCount = 0;

    const appendMessage = (text, sender) => {
        const msgDiv = document.createElement("div");
        msgDiv.classList.add("message", sender);
        msgDiv.innerHTML = text;
        messagesContainer.appendChild(msgDiv);
        chatWindow.scrollTop = chatWindow.scrollHeight;
    };

    const displayRecommendations = (recs) => {
        recommendationsContainer.innerHTML = "";
        recs.forEach(song => {
            const item = document.createElement("div");
            item.classList.add("recommendation-item");
            item.innerHTML = `
                <a href="${song.url}" target="_blank">
                    ${song.name} — ${song.artist}
                </a>
            `;
            recommendationsContainer.appendChild(item);
        });
        chatWindow.scrollTop = chatWindow.scrollHeight;
    };

    const sendMessage = () => {
        const userMessage = userInput.value.trim();
        if (!userMessage) return;
        appendMessage(userMessage, "user");
        userInput.value = "";

        if (chatStage === 1) {
            chatCount++;
            if (chatCount < 3) {
                // 正常聊天轮次
                fetch(`/get?msg=${encodeURIComponent(userMessage)}`)
                  .then(res => res.json())
                  .then(data => appendMessage(data.response, "bot"))
                  .catch(() => appendMessage("An error occurred. Please try again.", "bot"));
            } else {
                // 第 3 轮收尾
                const closing = "Thanks for sharing your thoughts! Let's use that to tailor your music recommendations.";
                appendMessage(closing, "bot");
                appendMessage("Please provide your Spotify playlist URL:", "bot");
                chatStage = 2;
            }
        }
        else if (chatStage === 2) {
            // 当做 Spotify URL 处理
            appendMessage("Got it! Fetching your playlist and recommendations...", "bot");
            loadingContainer.classList.remove("hidden");
            fetch(`/get?msg=${encodeURIComponent(userMessage)}`)
              .then(res => res.json())
              .then(data => {
                  loadingContainer.classList.add("hidden");
                  const recs = data.response;
                  if (Array.isArray(recs)) {
                      appendMessage("Here are some songs I recommend for you:", "bot");
                      displayRecommendations(recs);
                  } else {
                      appendMessage(data.response, "bot");
                  }
              })
              .catch(() => {
                  loadingContainer.classList.add("hidden");
                  appendMessage("An error occurred. Please try again.", "bot");
              });
            chatStage = 3;
        }
        else {
            // 推荐后依然可继续聊天
            fetch(`/get?msg=${encodeURIComponent(userMessage)}`)
              .then(res => res.json())
              .then(data => appendMessage(data.response, "bot"))
              .catch(() => appendMessage("An error occurred. Please try again.", "bot"));
        }
    };

    sendButton.addEventListener("click", sendMessage);
    userInput.addEventListener("keypress", e => {
        if (e.key === "Enter") sendMessage();
    });

    // Initial greeting
    appendMessage(
        "Hello! I am your personal music recommendation chatbot. I will chat with you to understand your mood in three dialogues. Let's begin! How are you today?",
        "bot"
    );
});
