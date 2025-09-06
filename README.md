# 🌱 AgriTime AI : Precision Planting Predictions for Small Farmers



**AgriTime AI** is a machine learning–based crop recommendation system that helps farmers and agricultural experts make better **data-driven decisions**.  
It analyzes soil and environmental parameters to recommend the most suitable crop for cultivation.

---

## ✨ Features
- 📊 Analyze soil and atmospheric parameters: **Nitrogen, Phosphorus, Potassium (NPK), pH, Humidity, Temperature, Rainfall**  
- 🌾 Predict the **best crop** to grow based on input data  
- 🤖 **Model Evaluation with 10 ML Algorithms**:
  - Logistic Regression  
  - Gaussian Naive Bayes (GaussianNB)  
  - Support Vector Classifier (SVC)  
  - K-Nearest Neighbors Classifier (KNN)  
  - Decision Tree Classifier  
  - Extra Tree Classifier  
  - Random Forest Classifier  
  - Bagging Classifier  
  - Gradient Boosting Classifier  
  - AdaBoost Classifier  
- 🌐 Clean and simple **Flask web interface**  
- 🔧 Optimized model performance using **Bayesian Optimization** 

---

## 🛠️ Tech Stack & Requirements
- Python 
- Flask – Web framework  
- Scikit-learn – Machine learning  
- Pandas, NumPy, Matplotlib, Seaborn – Data handling & visualization  
- HTML/CSS – Frontend  

---

## 🚀 How to Run
1. Clone the repository
   git clone https://github.com/your-username/AgriTime_AI.git
   cd AgriTime_AI
2. Create a virtual environment
    python -m venv venv
    venv\Scripts\activate   # On Windows
    source venv/bin/activate   # On Linux/Mac
3. Update dataset path in app.py
   crop = pd.read_csv(r"C:\AgriTimeAI\Crop_recommendation.csv")
4. Install dependencies
   pip install flask pandas numpy matplotlib seaborn scikit-learn scikit-optimize
5. Run the application
   python app.py
   
## OUTPUT:
<img width="749" height="753" alt="image" src="https://github.com/user-attachments/assets/737c4769-26f6-4f54-9d95-8175656b1d10" />

fig: Backend Performance Metrics and Model Evaluation Console Output

<img width="669" height="1004" alt="image" src="https://github.com/user-attachments/assets/94633297-1067-4d24-bb10-d1753bd6ba45" />

fig: Initial Input Screen / User Data Collection Interface

---

## 📝 License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file in the repository root for details.

---

**Created by  Laxmidhar Penta**




