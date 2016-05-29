from flask import Flask, request, jsonify
import numpy as np
import os, json
# set the project root directory as the static folder, you can set others.
root = os.path.join(os.path.dirname(os.path.abspath(__file__)))
print root

app = Flask(__name__, static_url_path='')

@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r

@app.route('/')
def root():
    return app.send_static_file('index.html')

@app.route('/answer', methods=['GET', 'POST'])
def get_answer():
    data = json.loads(request.form['data'])
    sentences = data['sentences']
    question = data['question']

    memory_probabilities = np.round(np.random.uniform(0, 1, (len(sentences), 3)), 4)

    response = {
        "answer": "test" + str(np.random.randint(100)),
        "answerProbability": round(np.random.uniform(), 3),
        "memoryProbabilities": memory_probabilities.tolist()
    }

    return jsonify(response)

if __name__ == "__main__":
    app.run()
