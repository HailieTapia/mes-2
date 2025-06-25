from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Cargar el modelo entrenado
with open('modelo_Random.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        # Obtener los datos del formulario
        displacement = float(request.form['displacement'])
        weight = float(request.form['weight'])
        model_year = float(request.form['model_year'])
        horsepower = float(request.form['horsepower'])
        
        # Preparar los datos en el formato esperado por el modelo
        input_data = np.array([[displacement, weight, model_year, horsepower]])
        
        # Hacer la predicción
        prediction = model.predict(input_data)[0]
    
    return render_template('index.html', prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)