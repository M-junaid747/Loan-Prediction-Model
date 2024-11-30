from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the pre-trained model
model = pickle.load(open('random_forest_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the JSON data from the request
    data = request.get_json(force=True)

    # Extract and convert the input data from the form
    no_of_dependents = int(data['no_of_dependents'])
    education = 0 if data['education'] == '0' else 1  # 0: Graduate, 1: Not Graduate
    self_employed = 1 if data['self_employed'] == '1' else 0  # 1: Yes, 0: No
    income_annum = float(data['income_annum'])
    loan_amount = float(data['loan_amount'])
    loan_term = int(data['loan_term'])
    cibil_score = float(data['cibil_score'])
    residential_assets_value = float(data['residential_assets_value'])
    commercial_assets_value = float(data['commercial_assets_value'])
    luxury_assets_value = float(data['luxury_assets_value'])
    bank_asset_value = float(data['bank_asset_value'])

    # Create a 2D array (with 1 row) of all input features as expected by the model
    features = np.array([[no_of_dependents, education, self_employed, income_annum, loan_amount, loan_term,
                          cibil_score, residential_assets_value, commercial_assets_value, luxury_assets_value, 
                          bank_asset_value]])

    # Make the prediction using the model
    prediction = model.predict(features)

    # Return the prediction as a JSON response
    result = 'Approved' if prediction[0] == 1 else 'Rejected'  # 1: Approved, 0: Rejected
    return jsonify(prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
