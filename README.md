# AgriTime_AI

This project uses machine learning to analyze soil parameters and recommend the most suitable crop for cultivation. It is aimed at helping farmers and agricultural experts make better data-driven decisions.

Features:
Analyze soil and atmosphere parameters like Nitrogen, Phosphorus, Potassium, pH, Humidity, Temperature, Rainfall.
Predict the best crop to grow based on the input data.
Clean and simple web interface using Flask.
Trained using a Naive Bayes model.

 Software Requirements:
.Python
.Flask for web framework
.Scikit-learn for ML
.HTML/CSS for frontend

How to Run:
.Create Virutal Environment
           python -m venv venv
           venv\Scripts\activate
.Update the Crop_recommendation.csv file path in app.py file
           crop = pd.read_csv(r"C:\AgriTimeAI\Crop_recommendation.csv")
.Install the required libraries using the command:
            pip install flask pandas numpy matplotlib seaborn scikit-learn scikit-optimize
.Run the app.py file:
         python app.py

