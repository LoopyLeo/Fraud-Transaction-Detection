# 💳 Fraud Transaction Detection using Machine Learning

## 📌 Overview
This project focuses on detecting fraudulent transactions using machine learning techniques. The dataset contains transaction-level data with features such as transaction time and amount. The goal is to accurately identify fraudulent transactions while minimizing false positives.

---

## 🎯 Objectives
- Detect fraudulent transactions from a highly imbalanced dataset  
- Build a reliable and realistic machine learning model  
- Improve precision, recall, and F1-score using advanced techniques  
- Apply threshold tuning for better fraud detection  

---

## 📊 Dataset
Due to large size, the dataset is not included in this repository.

👉 Download dataset from Kaggle:  
https://kaggle.com/datasets/2f520a15d4616921139b2da43f116d5e4645f3b754856aa15be8455914c4d4e4  

The dataset contains features such as:
- Transaction Time (seconds and days)  
- Transaction Amount  
- Fraud Label (`TX_FRAUD`)  

---

## ⚙️ Technologies Used
- Python  
- Pandas, NumPy  
- Scikit-learn  
- XGBoost  
- Matplotlib, Seaborn  

---

## 🔄 Workflow
1. Data Loading from multiple `.pkl` files  
2. Data Cleaning and Preprocessing  
3. Removal of data leakage features  
4. Handling class imbalance  
5. Model training using XGBoost  
6. Threshold tuning for optimal performance  
7. Evaluation using classification metrics and ROC/PR curves  

---

## 🤖 Model Used
- **XGBoost Classifier**  
- Optimized using hyperparameter tuning and class imbalance handling  

---

## 📈 Performance Metrics
- Accuracy: ~99%  
- Precision: ~58%  
- Recall: ~32%  
- F1 Score: ~41%  

The model achieves a good balance between detecting fraud and minimizing false positives.

---

## 📊 Evaluation
- Confusion Matrix  
- ROC Curve & AUC Score  
- Precision-Recall Curve  

---

## 🚀 Key Features
- Leakage-free pipeline  
- Handles imbalanced dataset effectively  
- Threshold optimization for better decision-making  
- Realistic fraud detection performance  

---

## ⚠️ Limitations
- Moderate recall (some fraud cases may be missed)  
- Dataset is synthetic, may differ from real-world scenarios  

---

## 🔮 Future Improvements
- Use deep learning models (LSTM, Autoencoders)  
- Real-time fraud detection system  
- Feature engineering for better recall  
- Ensemble models for improved performance  

---

## 👨‍💻 Author
Nikhil Anand  

---

## 📜 License
This project is for educational purposes.
