
**Mod Disease Detection and Health Care Recommendation System**

**Project Overview:**

This project delivers a machine learning-powered web application that enables users to:
📝 Input symptoms using natural language (e.g., "I have a sore throat and chills").
🧠 Receive predictions for potential illnesses using an ML model.
💊 Access recommendations for treatment, risk levels, and doctor specialties.

It’s designed to help users make early, informed healthcare decisions through a simple, intelligent web interface.
**Project Structure:**

```
SE_ML_Model/
├── static/
│   ├── css/
│       └── styles.css         # CSS styling for the web interface
├── templates/
│   └── index.html             # HTML template with symptom input & results display
├── app.py                     # Flask web server and routing logic
├── model.py                   # ML model training and prediction logic
├── dataset.csv                # Dataset used for training the ML model
└── requirements.txt           # Required Python packages

```

**Requirements:**

Ensure the following packages are installed:

bash
Copy
Edit
pip install -r requirements.txt
Key Libraries:
Flask
pandas
scikit-learn
spaCy

And run: python -m spacy download en_core_web_sm
joblib (for saving/loading models)

⚠ If you're on Windows and get errors while installing spaCy or blis, install Microsoft C++ Build Tools first.

**Installation and Running:**

1. **Download the Project:** Obtain the project files, including `requirements.txt`.
2. **Clone or Download the Repository**
      git clone https://github.com/yourusername/SE_ML_Model.git
      cd SE_ML_Model
3. **CCreate Virtual Environment (Optional but Recommended)**
   - Create a virtual environment to isolate project dependencies (e.g., using `venv` or `conda`):
     ```bash
     python -m venv my_env  # Create a virtual environment named 'my_env'
     source my_env/bin/activate  # Activate the virtual environment
     ```
   - Install required packages within the virtual environment:
     ```bash
     pip install -r requirements.txt
     ```
3. **Run the Application:** Open a terminal in the project directory (where `app.py` is located) and execute:
   ```bash
   python app.py
   ```

4. **Access the Web Interface:** Navigate to `http://127.0.0.1:5000/` in your web browser. You should see the application's symptom input field.

**Using the System:**


1. Type symptoms in plain English like:
2. I’ve been sneezing with chills and a sore throat.
3. The system will:
4. Extract valid symptoms using NLP (spaCy).
5. Predict the most likely disease.

Show:

Disease name
Confidence/probability
Recommended treatment
Doctor specialty
Risk level

**Features Added with NLP**

✅ Natural language symptom extraction using spaCy
✅ Automatic keyword matching with dataset symptom list
✅ Cleaned input improves prediction accuracy
✅ Fallbacks for unrecognized input
✅ Scalable NLP-ready design

**UI Screenshots**
C:\Users\USER\Desktop\dis\Disease_Detection_-_Health_Care_Recommendation-main\S E_ML_Model\static\screenshots

**Important Note:**

- This is a simplified example for demonstration purposes. Real-world healthcare applications require extensive medical expertise and data, and should not be used for self-diagnosis. Always consult a licensed medical professional for proper diagnosis and treatment.


Remember, this project provides a basic framework. Responsible development with thorough medical knowledge, data security, and ethical considerations is crucial for real-world healthcare applications.
