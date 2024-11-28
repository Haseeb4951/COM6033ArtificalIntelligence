from flask import Flask, render_template, request
import joblib
import numpy as np
from sklearn.metrics import mean_squared_error  

app = Flask(__name__)

model = joblib.load('gradient_boosting_model.pkl')
scaler = joblib.load('scaler.pkl')  

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [float(x) for x in request.form.values()]
        
        print("Parsed Features:", features)

        if len(features) != 8:
            return render_template('index.html', prediction_text="Error: Please provide all 8 features.")

        features = np.array(features).reshape(1, -1)

        scaled_features = scaler.transform(features)

        print("Scaled Features for Model:", scaled_features)

        prediction = model.predict(scaled_features)

        print("Model Prediction:", prediction)

        expected_value = 0

        mse = mean_squared_error([expected_value], prediction)

        print("Mean Squared Error (MSE):", mse)

        prediction_text = f'Predicted Diabetes Progression: {prediction[0]:.2f}'
        mse_text = f'Mean Squared Error (MSE): {mse:.4f}'

        return render_template('index.html', prediction_text=prediction_text, mse_text=mse_text)

    except Exception as e:
        print("Error:", str(e))
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
