from flask import Flask, request, jsonify
import pandas as pd
import pickle
from flask_cors import CORS  # Import CORS from flask_cors

app = Flask(__name__)
CORS(app)
# Load the model and preprocessor from the file using pickle
with open('fertilizer_prediction.pkl', 'rb') as model_file:
    loaded_model, loaded_preprocessor = pickle.load(model_file)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the JSON data from the request
        request_data = request.get_json()

        # Convert the JSON data to a DataFrame
        single_data_df = pd.DataFrame(request_data)

        # Transform the single test data point using the loaded preprocessor
        scaled_single_data = loaded_preprocessor.transform(single_data_df)

        # Make predictions using the loaded model
        predicted_label = loaded_model.predict(scaled_single_data)

        # Return the predicted label as JSON response
        response = {'predicted_label': predicted_label[0]}
        return jsonify(response)
    
    except Exception as e:
        # Return error response if an exception occurs
        error_message = str(e)
        response = {'error': error_message}
        return jsonify(response), 500  # HTTP status code 500 for internal server error

if __name__ == '__main__':
    app.run(debug=True)
