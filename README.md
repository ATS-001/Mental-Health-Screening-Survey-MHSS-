# 📋 Mental Health Screening Survey (MHSS)

An efficient, lightweight machine learning classification dashboard designed to process multi-dimensional wellness metrics, vectorize natural language text representations, and evaluate potential indicators of emotional distress with live probabilistic confidence tracking.

🚀 **Live Demo:** [Launch App on Streamlit Community Cloud](https://mhss-ats001.streamlit.app/) 

---

## 📅 Workshop Context

* **Event:** Day 4 of Projectathon conducted by μLearn LBSITW (29th June 2026)
* **Presented by:** Aiswarya Jayaprakash, Data Science IG LEAD, µLearn LBSITW
* **Focus:** Deep dive into supervised machine learning workflows, text normalization regex patterns, TF-IDF feature space extraction, and predictive confidence boundaries via Logistic Regression.

---

## ✨ Features
* **10-Dimensional Screening Array:** Multi-variant survey architecture covering clinical mood parameters, physical stamina drop indicators, social avoidance metrics, insomnia patterns, and cognitive focus shifts.
* **Integrated Natural Language Processing:** Collects open-ended text narratives alongside categorical options, collapsing structured inputs and prose blocks into an aggregated linguistic context window.
* **Vectorized Feature Extraction:** Uses a customized **TF-IDF Vectorizer** to transform high-dimensional raw content into numerical space tokens instantly.
* **Local In-Memory Inference:** Leverages an optimized **Logistic Regression Model** serialized directly into a `tfidf_model.pkl` binary to predict risk markers without dependancy on cloud APIs.
* **Isolated Execution Sandbox:** Built using isolated Python environment wrappers (`.venv`) to protect local system configurations from third-party binary updates.

---

## 🛠️ Architecture & Tech Stack

* **Frontend Framework:** Streamlit (Python-driven reactive web UI)
* **Model Serialization Architecture:** Pickle Framework (`tfidf_model.pkl`)
* **Core Machine Learning Toolkit:** Scikit-Learn (`scikit-learn==1.5.0`)
* **Data Engineering Stack:** Pandas (`pandas==2.2.2`) & Numpy
* **Linguistic Preprocessing:** Regular Expressions Engine (`re`)
* **Repository Architecture:** GitHub Cloud Ecosystem

---

## 🚀 Local Setup & Installation

If you want to run this application locally on your machine, follow these steps:

### 1. Clone the Repository

```bash
git clone [https://github.com/your-username/your-mhss-repo.git](https://github.com/your-username/your-mhss-repo.git)
cd your-mhss-repo
```

### 2. Configure Virtual Environment Sandbox

Initialize and link your local tracking variables to avoid dependency version collision:
```bash
# On Windows
python -m venv .venv
.venv\Scripts\activate

# On macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install System Dependencies

Ensure you have your environment activated, then install the package requirements wrapper:

```bash
pip install -r requirements.txt
```

### 4. Execute the Application Terminal Engine

```bash
.\.venv\Scripts\python.exe -m streamlit run app.py
```

---

## 🔮 Acknowledgments

Special thanks to Aishwarya Jayaprakash ([Github](https://github.com/Aiswarya-Jayaprakash)) for providing the data science infrastructural guidelines, foundational preprocessing rules, and the hands-on instruction that enabled this end-to-end classification system design and deployment.

---

### 👨‍💻 Developer Profile
* **Name:** Aaron Thalakkottor Sooraj
* **Degree:** B.Tech in Computer Science Engineering (CSE)
* **Institution:** Vidya Academy of Science and Technology, Thrissur

---

### 📜 License
```text
COPYRIGHT © Since 2023 ATS-PDZ - ALL RIGHTS RESERVED.
```
