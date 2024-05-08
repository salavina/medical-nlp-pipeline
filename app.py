from medical_nlp.config.configuration import configurationManager
from medical_nlp.components.model_training import ModelTrainerHF
from flask_cors import CORS, cross_origin
import os
from flask import Flask, request, render_template, jsonify


os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)


class MedicalApp:
    def __init__(self):
        config = configurationManager()
        training_config = config.get_training_config()
        self.classifier = ModelTrainerHF(config=training_config)


@app.route("/", methods=['GET', 'POST'])
@cross_origin()
def home():
    return render_template('index.html')
    # return '''
    #           <form method="POST" action="/predict" enctype="multipart/form-data">
    #               <div><label>Medical Text: <input type="text" name="medical"></label></div>
    #               <input type="submit" value="Submit">
    #           </form>'''


@app.route("/train", methods=['GET', 'POST'])
@cross_origin()
def trainRoute():
    os.system("dvc repro")
    return jsonify("Training Done Successfully!")


@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
   text = request.json['text']
   prediction = claApp.classifier.predict(text)
   return jsonify(prediction)

if __name__ == "__main__":
    claApp = MedicalApp()
    
    
    app.run(host='0.0.0.0', port=8080) #for AWS & local host
    # app.run(host='0.0.0.0', port=80) #for Azure