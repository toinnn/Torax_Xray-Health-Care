from flask import Flask

app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/oi')
def ola_world():
    return 'Olá, Mundo!'


if __name__ == '__main__':
    app.run(debug=True)


# python app.py
