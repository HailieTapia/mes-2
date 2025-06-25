from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Cargar el modelo entrenado con joblib
model = joblib.load('modelo_Random.pkl')

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        displacement = float(request.form['displacement'])
        weight = float(request.form['weight'])
        model_year = float(request.form['model_year'])
        horsepower = float(request.form['horsepower'])
        
        input_data = np.array([[displacement, weight, model_year, horsepower]])
        
        prediction = model.predict(input_data)[0]
    
    return render_template('index.html', prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)