from flask import Flask, request, render_template
from gensim.utils import simple_preprocess
from transformers import CamembertTokenizer, AutoTokenizer
from transformers import pipeline, TFCamembertForSequenceClassification, TFAutoModelForSequenceClassification
import fasttext
# import torch


positive_review = "Magnifique épopée, une belle histoire, touchante avec des acteurs qui interprètent très bien leur rôles (Mel Gibson, Heath Ledger, Jason Isaacs...), le genre de film qui se savoure en famille! :)"
negative_review = "Un scenario indigent qu'on étire comme un vieux chewing-gum. Des images de la Cote basque. Une mise en scène sans intérêt. Les visages de la famille Attal-Gainsbourg en gros plans répétitifs. Certaines scènes très mal jouées. On l'a compris c'est plus une purge qu'un film."

app = Flask(__name__)

# tokenizer takes time too
tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
# # using transformers from Huggingface 
# tokenizer = AutoTokenizer.from_pretrained("tblard/tf-allocine")

# loading models takes time
model = TFCamembertForSequenceClassification.from_pretrained("jplu/tf-camembert-base")
model.load_weights('models/camembert_weights.hdf5')
# model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
# model = TFAutoModelForSequenceClassification.from_pretrained("tblard/tf-allocine")


@app.route('/')
def my_form():
    return render_template('form.html')

@app.route('/', methods=['POST'])
def my_form_post():
    
    #get text data
    text1 = request.form['text1']
        
    ########################Fasttext inference########################
    # # process it for inference
    # cleand_text =  simple_preprocess(text1)
    # cleand_text = ' '.join([word for word in cleand_text])
    
    # model = fasttext.load_model("models/fasttext_model_quant.ftz")
    # pred = model.predict(text1)

    #######################Camembert inference########################    
    # create pipeline and perform inference
    nlp = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)
    pred = nlp(text1)

    return render_template('form.html', text1=text1, label=pred[0]['label'], score=round(pred[0]['score'], 4))

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5002, threaded=True)