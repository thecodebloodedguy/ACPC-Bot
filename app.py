from flask import Flask, request, jsonify,render_template
app=Flask(__name__)
from chatbot import reply,answer_lines,lines


@app.route("/", methods=['GET', 'POST'])
def home():
    response = None
    if request.method == 'POST':
        user_input = request.form['user_input']

        # Implement your chatbot logic here
        # For this example, we'll use the percent_matching function as an example
        response = reply(user_input)

    return render_template('index.html', response=response)


if __name__ == '__main__':
    app.run(debug=True)



