from flask import Flask, request, render_template, flash
import numpy as np
import pickle

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Required for flashing messages
model = pickle.load(open("crop2.pkl", 'rb'))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    try:
        # Get form data
        int_features = [float(x) for x in request.form.values()]
        N, P, K, temperature, humidity, ph, rainfall = int_features

        # Server-side validation
        if N < 0 or P < 0 or K < 0 or rainfall < 0:
            flash("Negative values are not allowed for Nitrogen, Phosphorus, Potassium, or Rainfall.")
            return render_template("index.html")

        if ph < 0 or ph > 14:
            flash("pH value must be between 0 and 14.")
            return render_template("index.html")

        if humidity < 0 or humidity > 100:
            flash("Humidity must be between 0% and 100%.")
            return render_template("index.html")

        # If all validations pass, proceed with prediction
        final_features = [np.array(int_features)]
        prediction = model.predict(final_features)
        res = str(prediction)[1:-1]  # Remove brackets from prediction

        return render_template("index.html", prediction_text="{}".format(res))

    except Exception as e:
        flash("An error occurred while processing your request. Please check your inputs.")
        return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)