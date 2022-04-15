#!/usr/bin/python

import json
from flask import Flask, render_template, redirect, url_for, request
from flask_wtf import FlaskForm
from requests.exceptions import ConnectionError
from wtforms import IntegerField, SelectField, StringField
from wtforms.validators import DataRequired

import os
import urllib.request

class ClientDataForm(FlaskForm):
    title = StringField('title', validators=[DataRequired()])
    content = StringField('content', validators=[DataRequired()])


app = Flask(__name__)
app.config.update(
    CSRF_ENABLED=True,
    SECRET_KEY='qwerty',
)

def get_prediction(title, content):
    body = {'title': title, 'content': content}

    myurl = "http://127.0.0.1:8180/predict"
    req = urllib.request.Request(myurl)
    req.add_header('Content-Type', 'application/json; charset=utf-8')
    jsondata = json.dumps(body)
    jsondataasbytes = jsondata.encode('utf-8')   # needs to be bytes
    req.add_header('Content-Length', len(jsondataasbytes))
    response = urllib.request.urlopen(req, jsondataasbytes)
    return json.loads(response.read())['predictions']

@app.route("/")
def index():
    return render_template('index.html')


@app.route('/predicted/<response>')
def predicted(response):
    response = json.loads(response)
    print(response)
    return render_template('predicted.html', response=response)


@app.route('/predict_form', methods=['GET', 'POST'])
def predict_form():
    form = ClientDataForm()
    data = dict()
    if request.method == 'POST':
        data['title'] = request.form.get('title')
        data['content'] = request.form.get('content')


        try:
            response = str(get_prediction(data['title'],
                                      data['content']))
            print(response)
        except ConnectionError:
            response = json.dumps({"error": "ConnectionError"})
        return redirect(url_for('predicted', response=response))
    return render_template('form.html', form=form)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8181))
    app.run(host='127.0.0.1', port=port, debug=True)
