from flask import Flask, render_template, request
from chatbot import get_answer

app = Flask(__name__)

chat_history = []

@app.route("/", methods=["GET", "POST"])
def home():

    if request.method == "POST":

        user_question = request.form["question"]

        answer = get_answer(user_question)

        chat_history.append(
            {
                "question": user_question,
                "answer": answer
            }
        )

    return render_template(
        "index.html",
        chat_history=chat_history
    )

if __name__ == "__main__":
    import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)