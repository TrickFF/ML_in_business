#!/usr/bin/python

# Импортируем библиотеки
import flask
import os
import re
from joblib import load
import pymorphy2
import warnings

# Инициализация Flask и модели
app = flask.Flask(__name__)
vectorizer = None
model = None

# Список слов исключений на русском
with open('text.txt') as f:
    stopword_ru = [w.strip() for w in f.readlines() if w]

# Список слов исключений на английском
with open('text_eng.txt') as f:
    exclude_words_eng = [w.strip() for w in f.readlines() if w]

# Загружаем векторизатор и модель
def load_model(modelpath, vectorizerpath):
    global vectorizer
    global model
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        vectorizer = load(vectorizerpath)
        model = load(modelpath)

# Очищаем текст
def cleanhtml(raw_text):
    
    if not isinstance(raw_text, str):
        raw_text = str(raw_text)
    
    raw_text = raw_text.lower()
    cleantext = raw_text.split('уважением')[0]
    cleantext = cleantext.split('с пожеланием')[0]
    cleantext = cleantext.split('с наилучшими')[0]
    cleantext = cleantext.split('заранее спасибо')[0]
    
    if cleantext.find('здравствуйте') != -1:
        cleantext = cleantext.split('здравствуйте')[1]
    
    if cleantext.find('здравствуй') != -1:
        cleantext = cleantext.split('здравствуй')[1]
        
    if cleantext.find('добрый день') != -1:
        cleantext = cleantext.split('добрый день')[1]

    if cleantext.find('добрый вечер') != -1:
        cleantext = cleantext.split('добрый вечер')[1]

    if cleantext.find('доброе утро') != -1:
        cleantext = cleantext.split('доброе утро')[1]

    if cleantext.find('приветствую') != -1:
        cleantext = cleantext.split('приветствую')[1]     
    
    cleantext = cleantext.strip('\n').strip('\r').strip('\t')
    cleantext = re.sub("-\s\r\n\|-\s\r\n|\r\n", ' ', str(cleantext))
    cleantext = re.sub("[-—}{.,:;_%©«»?*!@#№$^•·&()+=]", ' ', cleantext)
    cleantext = re.sub(r"\r\n\t|\n|\\s|\r\t|\\n", ' ', cleantext)
    cleantext = re.sub(r'[\xad]|[\s+]', ' ', cleantext.strip())
  
    cleantext = cleantext.replace(u'\u200e', ' ')   
    cleantext = cleantext.replace('"', ' ')
    cleantext = cleantext.replace('“', ' ')
    cleantext = cleantext.replace('”', ' ')
    cleantext = cleantext.replace("'", ' ')    
    cleantext = cleantext.replace('[', ' ')
    cleantext = cleantext.replace(']', ' ')
    cleantext = cleantext.replace('\\', ' ')
    cleantext = cleantext.replace('//', ' ')
    cleantext = cleantext.replace('/', ' ')
    cleantext = cleantext.replace('…', ' ')
    cleantext = cleantext.replace('nbsp', ' ')

    return cleantext

# Леммаизация текста
def lemmatization(text):
    
    morph = pymorphy2.MorphAnalyzer()

    if not isinstance(text, str):
        text = str(text)
    
    words = text.split(' ')
    words_without_digits_and_spaces = [i for i in words if i.isdigit() == False and len(i) > 1 and int(bool(re.search('[а-яА-Я]', i)) == True or i in exclude_words_eng) >= 1]  
    words_without_stopwords = [i for i in words_without_digits_and_spaces if not i in stopword_ru]   
    words_lemm = [morph.parse(i)[0].normal_form for i in words_without_stopwords if i != '1c' or i != '1с']
    
    return words_lemm

# Обрабатываем данные
def data_processing(title, content):
    
    result = ''    
    content = cleanhtml(content)
    title = cleanhtml(title)
    content = lemmatization(content)
    title = lemmatization(title)

    for _ in range(len(title)):
        content.append(title[_])

    for el in content:
        result += f' {el}'
    
    return result

@app.route("/predict", methods=["POST"])
def predict():
    # Инициализация словаря в котором будет отправлен ответ от сервера
    data = {"success": False}

    # Проверяем загрузку
    if flask.request.method == "POST":
        title, content = "", ""
        request_json = flask.request.get_json()
        
        # Проверяем содержание полученных полей
        # if request_json["title"]:
        title = request_json['title']
        # if request_json["content"]:
        content = request_json['content']
        
        # Пробуем получить предсказание класса
        try:
            text = [data_processing(title, content)]
            text =  vectorizer.transform(text)
            preds = model.predict(text)
        # Если что-то пойдет не так, то в ответ будет отправлена ошибка
        except AttributeError as e:
            data["predictions"] = str(e)
            data["success"] = True
            return flask.jsonify(data)
        
        # Если все прошло корректно - заносим данные в словарь
        data["predictions"] = int(preds)
        data["success"] = True

    # Возвращаем полученное предсказание
    return flask.jsonify(data)

# Основной блок запуска сервера
if __name__ == "__main__":
    vectorizerpath = "./models/vectorizer.pkl"  
    modelpath = "./models/model.pkl"
    load_model(modelpath, vectorizerpath)
    port = int(os.environ.get('PORT', 8180))
    app.run(host='127.0.0.1', debug=True, port=port)
