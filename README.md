# Query Management System with Semantic Search

A Flask-based intelligent system to manage and retrieve answers to queries using semantic search powered by BERT. Users can upload questions (with or without answers) via Excel or CSV files, and later search for similar past questions and their answers.

## 🎯 Use Case

Perfect for organizations handling large volumes of policy, legal, procurement, or public-related questions where historical answers can help reduce redundant effort.

## 🚀 Features

- ✅ Role based access: Admin, Modifier, Viewer
- ✅ Upload questions via Excel/CSV
- ✅ Search by question, reply, memo ID, or state name
- ✅ Semantic search using MiniLM (BERT-based)
- ✅ Admin approval for new users
- ✅ Answers can be added later for unanswered queries
- ✅ Download search results as Excel
- ✅ Dashboard: total questions, unanswered questions, state-wise statistics
- ✅ Secure login & session management
- ✅ Activity logging for audit

## 🖥️ Tech Stack

- **Backend**: Flask, Python
- **Database**: PostgreSQL (via SQLAlchemy)
- **Frontend**: HTML, Bootstrap
- **AI/NLP**: HuggingFace Transformers (MiniLM)
- **File Handling**: pandas, openpyxl

## 📁 Folder Structure

query-management-system/
├── app.py
├── templates/
├── static/
├── uploads/
├── logs/
├── .env.example
└── requirements.txt

## ⚙️ Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/joannaeliza222/Query_Management_System.git
cd query-management-system

### 2. Set up virtual environment
python -m venv venv
source venv/bin/activate

### 3.Install dependencies
pip install -r requirements.txt

### 4. Create and configure .env file
cp .env.example .env  (Fill in your actual database URL, secret key, etc.)

### 5.Run the application
python app.py
