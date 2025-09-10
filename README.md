
# 📊 Data Pre-Processing Project

A **Django-based web application** that automates **data cleaning, preprocessing, and machine learning model selection**.  

---

## 🚀 Features

### Data Upload & Cleaning
- Upload CSV files directly from the browser.  
- Handle **missing values** automatically.  
- Detect and treat **outliers**.  

### Model Training & Selection
- Train multiple models (e.g., Linear Regression, Random Forest).  
- Automatically select the **best model** based on evaluation metrics.  
- Save and download models in `.joblib` format.  

### User Interface
- Simple **Django template-based UI** (HTML, CSS).  
- Results displayed clearly for quick analysis.  

---

## 🏗️ Project Structure

Data-Pre-Processing-Project/
├── myproject/                # Django project settings
│   ├── settings.py
│   ├── urls.py
│   └── …
├── myapp/                    # Core application logic
│   ├── views.py              # Handles requests & ML pipeline
│   ├── models.py             # Database models (if used)
│   ├── forms.py              # File upload forms
│   └── templates/            # Frontend UI (HTML)
├── media/                    # Uploaded datasets
├── best_model_*.joblib       # Trained models (exported)
├── db.sqlite3                # SQLite development database
└── manage.py                 # Django CLI entry point

---

## ⚙️ Tech Stack

- **Backend:** Django (Python), Scikit-learn  
- **Frontend:** HTML, CSS, Django Templates  
- **Database:** SQLite (default)  
- **Model Storage:** Joblib  

---

## 🛠️ Setup & Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/mansoorsaadmohammed2003/Data-Pre-Processing-Project.git
   cd Data-Pre-Processing-Project

	2.	Create & activate virtual environment:

python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate


	3.	Install dependencies:

pip install -r requirements.txt


	4.	Apply migrations:

python manage.py migrate


	5.	Run the server:

python manage.py runserver


	6.	Open in browser:

http://127.0.0.1:8000/



⸻

📂 Workflow
	1.	Upload dataset (.csv).
	2.	System cleans data (missing values, outliers).
	3.	Multiple ML models trained & compared.
	4.	Best model auto-selected & stored (joblib).
	5.	User can download the trained model.

⸻

🤝 Contributing
	1.	Fork the repo.
	2.	Create a branch (git checkout -b feature-name).
	3.	Commit changes (git commit -m 'Added feature').
	4.	Push to branch (git push origin feature-name).
	5.	Open a Pull Request.


