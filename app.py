from flask import Flask, request, jsonify,render_template
app=Flask(__name__)
from chatbot import reply,answer_lines,lines



# Simple rule-based chatbot
def chatbot_response(user_input):
    print(reply(user_input))
    return reply(user_input)


@app.route("/", methods=["GET", "POST"])
def chatbot_interface():
    chat_history = []

    if request.method == "POST":
        user_input = request.form["user_input"]
        user_message = {"sender": "user", "content": user_input}
        chat_history.append(user_message)

        bot_response = chatbot_response(user_input)
        bot_message = {"sender": "bot", "content": bot_response}
        chat_history.append(bot_message)

    return render_template("index.html", chat_history=chat_history)



# @app.route("/", methods=['GET', 'POST'])
# def home():
#     response = None
#     if request.method == 'POST':
#         user_input = request.form['user_input']

#         # Implement your chatbot logic here
#         # For this example, we'll use the percent_matching function as an example
#         response = reply(user_input)

#     return render_template('index.html', response=response)


if __name__ == '__main__':
    app.run(debug=True)



