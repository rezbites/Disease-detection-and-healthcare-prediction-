from flask import Flask, render_template, request
from model import predict_disease, get_disease_details, extract_symptoms_from_text

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    details = None
    if request.method == 'POST':
        raw_input = request.form['symptoms']
        
        extracted = extract_symptoms_from_text(raw_input)

        if not extracted:
            details = {
                'Symptoms': raw_input,
                'Disease': 'Could not recognize symptoms.',
                'Cure': 'N/A',
                'Doctor': 'N/A',
                'Risk': 'N/A',
                'Probability': 'N/A'
            }
        else:
            symptoms_string = ','.join(extracted)
            top_disease, top_prob = predict_disease(symptoms_string)
            details = get_disease_details(top_disease)
            details['Probability'] = f"{top_prob:.4f}"
            details['Symptoms'] = ', '.join(extracted)

    return render_template('index.html', details=details)

if __name__ == "__main__":
    app.run(debug=True)
