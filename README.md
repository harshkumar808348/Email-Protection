Email Spam Detection System 📧🛡️
A machine learning-based email spam detection system that classifies emails as spam or legitimate (ham) using Logistic Regression with 96% accuracy.
Show Image
🚀 Features

High Accuracy: Achieves 96% accuracy in spam detection
Machine Learning Powered: Uses Logistic Regression algorithm
Text Processing: Advanced feature extraction using TF-IDF vectorization
Real-time Classification: Fast prediction on new email content
User-friendly Interface: Clean and intuitive web interface

🛠️ Technology Stack

Python 3.x
Scikit-learn: Machine learning library
Pandas: Data manipulation and analysis
NumPy: Numerical computing
TF-IDF Vectorizer: Text feature extraction
Logistic Regression: Classification algorithm

📊 Model Performance

Algorithm: Logistic Regression
Accuracy: 96%
Feature Extraction: TF-IDF (Term Frequency-Inverse Document Frequency)
Training Dataset: Email corpus with spam and ham labels

🔧 Installation

Clone the repository

bashgit clone https://github.com/harshkumar808348/Email-Protection.git
cd Email-Protection

Install required dependencies

bashpip install -r requirements.txt

Run the application

bashpython app.py
📋 Requirements
Create a requirements.txt file with:
scikit-learn==1.3.0
pandas==2.0.3
numpy==1.24.3
flask==2.3.2
nltk==3.8.1
matplotlib==3.7.2
seaborn==0.12.2
🎯 How It Works

Data Preprocessing: Clean and prepare email text data
Feature Extraction: Convert text to numerical features using TF-IDF
Model Training: Train Logistic Regression on labeled email data
Prediction: Classify new emails as spam or ham
Evaluation: Achieve 96% accuracy on test dataset

💻 Usage
Training the Model
pythonfrom sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load and preprocess data
X_train, X_test, y_train, y_test = train_test_split(emails, labels, test_size=0.2)

# Feature extraction
vectorizer = TfidfVectorizer(lowercase=True, stop_words='english')
X_train_features = vectorizer.fit_transform(X_train)

# Train model
model = LogisticRegression()
model.fit(X_train_features, y_train)
Making Predictions
python# Predict on new email
new_email = ["Congratulations! You've won $1000! Click here now!"]
email_features = vectorizer.transform(new_email)
prediction = model.predict(email_features)
print("Spam" if prediction[0] == 1 else "Ham")
📈 Model Metrics

Accuracy: 96%
Precision: High precision in spam detection
Recall: Effective at catching spam emails
F1-Score: Balanced performance metric

🔍 Dataset Information
The model is trained on a comprehensive email dataset containing:

Spam emails: Marketing emails, phishing attempts, scams
Ham emails: Legitimate personal and business communications
Features: Email subject lines, body content, metadata

🎨 Web Interface Features

Email input text area
Real-time spam/ham classification
Confidence score display
Clean, responsive design
Easy-to-use interface

🚦 Future Enhancements

 Deep learning models (LSTM, BERT)
 Email attachment analysis
 Multi-language support
 API integration
 Real-time email monitoring
 Advanced feature engineering

📝 Contributing

Fork the repository
Create your feature branch (git checkout -b feature/AmazingFeature)
Commit your changes (git commit -m 'Add some AmazingFeature')
Push to the branch (git push origin feature/AmazingFeature)
Open a Pull Request

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
👨‍💻 Author
Harsh Kumar

GitHub: @harshkumar808348

🙏 Acknowledgments

Scikit-learn documentation and community
Email spam detection research papers
Open source email datasets
Machine learning community
