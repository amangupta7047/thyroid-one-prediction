from flask import Flask, render_template, request
import requests
import pickle
import numpy as np
import sklearn

app = Flask(__name__)
model = pickle.load(open('tddmodel.pkl', 'rb'))
encoder = pickle.load(open('encoder.pickle', 'rb'))


@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    
    if request.method == 'POST':
        
        age = request.form['age']
        
        sex= request.form['sex']
        
        TSH = request.form['TSH']
        
        TT4 = request.form['TT4']
        
        FTI = request.form['FTI']
        
        T3 = request.form['T3']
        
        T4U = request.form['T4U']
        
        on_thyroxine = request.form['on_thyroxine']
        
        on_antithyroid_medication = request.form['on_antithyroid_medication']
        
        goitre = request.form['goitre']
        
        hypopituitary = request.form['hypopituitary']
        
        psych = request.form['psych']
        age=float(age)
        sex=float(sex)
        TSH=float(TSH)
        TT4=float(TT4)
        FTI=float(FTI)
        T3=float(T3)
        T4U=float(T4U)
        on_thyroxine=float(on_thyroxine)
        on_antithyroid_medication=float(on_antithyroid_medication)
        goitre=float(goitre)
        hypopituitary=float(hypopituitary)
        psych=float(psych)
        
        query = [age,sex,on_thyroxine,on_antithyroid_medication,goitre,hypopituitary,psych,TSH,T3,TT4,T4U,FTI]
        
        print(query)

        predicted_class = model.predict(np.array(query).reshape(1,12))[0]
        
        predicted_class = int(predicted_class)
    
        class_name = encoder.inverse_transform([predicted_class])[0]
        
        return render_template('index.html', prediction_text=class_name)
    
if __name__=="__main__":
    app.run(debug=True)



   