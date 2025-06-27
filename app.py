from flask import Flask,render_template,request, jsonify
import random
import pickle
import numpy as np
from openai import OpenAI

model = pickle.load(open('classifier1.pkl','rb'))
client = OpenAI(api_key = "ENTER YOUR OWN OPENAI API KEY") ; 

app = Flask(__name__)
# Initialize the OpenAI client (new way)

@app.route('/')
def index():
     return render_template('main.html')

@app.route('/bot')
def botfun():
     return render_template('bot.html')

@app.route('/weather')
def weatherfun():
     return render_template('weather.html')

@app.route('/fcalculator')
def fcalculatorfun():
     return render_template('fcalculator.html')


@app.route("/fpredictor")
def recommender():
    # Add your logic for the recommender app here
    # Return the appropriate template or data
    return render_template("fpredictor.html")

@app.route('/predict',methods=['POST'])
def predict():
    Nitrogen = (request.form.get('Nitrogen'))
    Potassium = (request.form.get('Potassium'))
    Phosphorous = (request.form.get('Phosphorous'))

    # prediction
    result = model.predict(np.array([[Nitrogen,Potassium,Phosphorous]]))
    # result = [1]
    # result[0] = random.randint(0,6)
    if result[0] == 0:
        result = 'TEN-TWENTY SIX-TWENTY SIX'
    elif result[0] == 1:
        result = 'Fourteen-Thirty Five-Fourteen'
    elif result[0] == 2:
        result = 'Seventeen-Seventeen-Seventeen'
    elif result[0] == 3:
        result = 'TWENTY-TWENTY'
    elif result[0] == 4:
        result = 'TWENTY EIGHT-TWENTY EIGHT'
    elif result[0] == 5:
        result = 'DAP'
    else:
        result = 'UREA'




    return render_template('fpredictor.html',result=str(result))

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_message = data['message']
        
        # Create a farming-focused system prompt
        system_prompt = """You are a knowledgeable farming assistant. Only provide information related to agriculture, 
        farming, crops, soil management, and related topics. If the question is not related to farming, 
        politely redirect the user to ask farming-related questions. Keep responses practical and accurate."""
        
        # Updated API call for OpenAI >= 1.0.0
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Using gpt-3.5-turbo instead of non-existent gpt-4o-mini
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.7,
            max_tokens=150
        )
        
        # Updated response parsing
        bot_response = response.choices[0].message.content
        return jsonify({'response': bot_response})
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'response': "I apologize, but I encountered an error. Please try again."})

if __name__ == '__main__':
    app.run(debug=True)
