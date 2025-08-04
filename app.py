from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
from sklearn.metrics.pairwise import cosine_similarity
from flask_sqlalchemy import SQLAlchemy
import torch
import pandas as pd
import os
from werkzeug.utils import secure_filename, send_file
from sqlalchemy import or_
from transformers import AutoTokenizer, AutoModel
from dotenv import load_dotenv
from waitress import serve
from datetime import datetime
import logging
import warnings
from openpyxl import Workbook, load_workbook
from sqlalchemy import or_
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash
from models import db, User
warnings.simplefilter(action='ignore', category=FutureWarning)

# Load environment variables from .env file
load_dotenv()

# Configure logging
if not os.path.exists('logs'):
    os.makedirs('logs')

logging.basicConfig(
    filename='logs/app.log', level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), 'templates'),
            static_folder=os.path.join(os.path.dirname(__file__), 'static'))

app.secret_key = os.getenv('SECRET_KEY', 'default_fallback_key')

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = os.getenv('UPLOAD_FOLDER', 'uploads')
app.config['ALLOWED_EXTENSIONS'] = {'xls', 'xlsx', 'csv'}

db = SQLAlchemy(app)

# Database Model

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    role = db.Column(db.String(50))
    state_name = db.Column(db.String(100))
    is_approved = db.Column(db.Boolean, default=False)
    
class FAQ(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    question = db.Column(db.String, unique=True, nullable=False)
    reply = db.Column(db.String, nullable=True)
    memo_id = db.Column(db.String, nullable=True)
    state_name = db.Column(db.String, nullable=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    __table_args__ = (db.UniqueConstraint('question', 'state_name', name='uq_question_state'),)

class PendingUser(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    requested_on = db.Column(db.DateTime, default=datetime.utcnow)
    
# Load BERT tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

df = pd.DataFrame(columns=["question", "reply", "memo_id", "state_name"])
df.to_excel("static/faq_template.xlsx", index=False)

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Admin Required Decorator
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        user = User.query.get(session['user_id'])
        if user.role != 'admin':
            flash("Admin access required")
            return redirect(url_for('home'))
        return f(*args, **kwargs)
    return decorated_function
    
def get_bert_embeddings(text):
    """Get BERT embeddings for a given text."""
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].numpy()


def find_related_questions(question, reply, memo_id, state_name):
    """Find related questions or replies from the database."""
    related_questions = []
    query = FAQ.query

    if memo_id:
        query = query.filter_by(memo_id=memo_id)
    if state_name:
        query = query.filter_by(state_name=state_name)
    if question or reply:
        query_text = question if question else reply
        query_embedding = get_bert_embeddings(query_text)
        faqs = query.all()

        similarity_scores = []
        
        for faq in faqs:

            db_text = faq.question if question else faq.reply
            db_embedding = get_bert_embeddings(db_text)
            similarity_score = cosine_similarity(query_embedding, db_embedding)[0][0]

            if similarity_score > 0.75:  # Adjust threshold
                similarity_scores.append((faq, similarity_score))
                #related_questions.append((faq.question, faq.reply, faq.memo_id, faq.state_name))
        
        # Sort related questions by similarity score (highest first)
        similarity_scores.sort(key=lambda x: x[1], reverse=True)
        related_questions = [(faq.question, faq.reply, faq.memo_id, faq.state_name) for faq, _ in
                             similarity_scores]
    else:
        faqs = query.all()
        for faq in faqs:
            related_questions.append((faq.question, faq.reply, faq.memo_id, faq.state_name))

    return related_questions


def fetch_data():
    with app.app_context():
        total_questions = FAQ.query.count()
        total_states = db.session.query(FAQ.state_name).distinct().count()
        unanswered_questions = FAQ.query.filter((FAQ.reply.is_(None)) | (FAQ.reply == '')).count()
        state_wise_count = db.session.query(FAQ.state_name, db.func.count(FAQ.id)).group_by(FAQ.state_name).all()
        return total_questions, total_states, unanswered_questions, state_wise_count

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


# Routes

@app.route('/', methods=['GET', 'POST'])
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, password):
            if user.is_approved:
                session['user_id'] = user.id
                return redirect(url_for('index'))
            else:
                flash("Awaiting admin approval")
                return redirect(url_for('login'))
        else:
            flash("Invalid credentials")
            return redirect(url_for('login'))
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        role = request.form['role']  # viewer or modifier
        state = request.form['state_name']
        hashed_pw = generate_password_hash(password)
        if User.query.filter_by(email=email).first():
            flash("Email already registered")
            return redirect(url_for('register'))
        new_user = User(email=email, password=hashed_pw, role=role, state_name=state)
        db.session.add(new_user)
        db.session.commit()
        flash("Registered successfully. Awaiting admin approval.")
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    session.clear()
    flash('Logged out successfully.', 'info')
    return redirect(url_for('login'))

@app.route('/index')
@login_required
def index():
    user = db.session.get(User, session['user_id'])
    total_questions, total_states, unanswered_questions, state_wise_count = fetch_data()
    if user.role == 'admin':
        pending_users = User.query.filter_by(is_approved=False).all()
        return render_template('index.html', user=user, pending_users=pending_users, total_questions=total_questions, total_states=total_states,
                           unanswered_questions=unanswered_questions, state_wise_count=state_wise_count,
                           username=session.get('username'), role=session.get('role'))

    return render_template('index.html', total_questions=total_questions, total_states=total_states,
                           unanswered_questions=unanswered_questions, state_wise_count=state_wise_count,
                           username=session.get('username'), role=session.get('role'), user=user)

# Route to display and manage pending questions
@app.route('/pending', methods=['GET', 'POST'])
@login_required
def pending():
    user = db.session.get(User, session['user_id'])
    role = user.role
    state_name = user.state_name

    if role == 'admin':
        states = [row[0] for row in db.session.execute(db.text("SELECT DISTINCT state_name FROM faq WHERE state_name IS NOT NULL"))]
    else:
        states = [state_name]
        
    if request.method == 'POST':
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                flash('No selected file', 'danger')
                return redirect(request.url)

            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)

                if filename.endswith('.csv'):
                    data = pd.read_csv(filepath)
                else:
                    data = pd.read_excel(filepath)

                if 'question' in data.columns:
                    data = data.dropna(subset=['question'])
                    merged_count = 0
                    duplicate_count = 0

                    for _, row in data.iterrows():
                        question = row.get('question', '').strip()
                        memo_id = row.get('memo_id', None)
                        state_name = row.get('state_name', None)

                        if question:
                            if not FAQ.query.filter_by(question=question, state_name=state_name).first():
                                new_entry = FAQ(question=question, memo_id=memo_id, state_name=state_name)
                                db.session.add(new_entry)
                                merged_count += 1
                            else:
                                duplicate_count += 1

                    db.session.commit()
                    flash(f'{merged_count} questions merged and {duplicate_count} duplicates found!', 'success')
                else:
                    flash('Invalid file format! Missing "question" column.', 'danger')
            else:
                flash('File type not allowed!', 'danger')


        if 'new_entry_id' in request.form and 'reply' in request.form:
            new_entry_id = request.form['new_entry_id']
            reply = request.form['reply']
            faq = FAQ.query.get(new_entry_id)
            if faq:
                faq.reply = reply
                db.session.commit()
                flash('Reply added successfully!', 'success')
            else:
                flash('FAQ not found!', 'danger')
        return redirect(url_for('pending'))

    # Fetch distinct state names for the dropdown
    distinct_states = FAQ.query.with_entities(FAQ.state_name).distinct().all()
    distinct_states = [state[0] for state in distinct_states if state[0]]  # Filter out None values

    # Handle state filter
    selected_state = request.args.get('state', '')
    # Fetch pending questions (questions without replies or with empty replies)
    query = FAQ.query.filter(or_(FAQ.reply.is_(None), FAQ.reply == ''))
    if role in ['modifier', 'viewer']:
        query = query.filter(FAQ.state_name == state_name)
    elif role == 'admin' and selected_state:
        query = query.filter(FAQ.state_name == selected_state)

    pending_questions = query.all()
    print("Pending Questions:", pending_questions) 

    return render_template('pending.html', questions=pending_questions, distinct_states=distinct_states,
                           selected_state=selected_state, states=states, username=user.email, role=role)


@app.route('/replied', methods=['GET', 'POST'])
@login_required
def replied():
    related_questions = []
    distinct_states = FAQ.query.with_entities(FAQ.state_name).distinct().order_by(FAQ.state_name).all()
    keyword = None
    
    if request.method == 'POST':
        user_question = request.form.get('question', '').strip()
        user_reply = request.form.get('reply', '').strip()
        memo_id = request.form.get('memo_id', '').strip()
        state_name = request.form.get('state_name', '').strip()
        keyword = request.form.get('keyword', '').strip()
        download = request.form.get('download', '') == 'true'

        if keyword:
            query = FAQ.query.filter(FAQ.reply.ilike(f"%{keyword}%"))
            related_questions = [(faq.question, faq.reply, faq.memo_id, faq.state_name) for faq in query.all()]
            print(f"Found {len(related_questions)} results for keyword: {keyword}")  

        elif user_question or user_reply or memo_id or state_name:
            related_questions = find_related_questions(user_question, user_reply, memo_id, state_name)
            if not related_questions:
                flash("No similar questions or replies found. Would you like to add this?", 'info')
        else:
            flash("Please enter a question, reply, memo ID, or state name to search.", 'warning')

        if download and related_questions:
            df = pd.DataFrame(related_questions, columns=["question", "reply", "memo_id", "state_name"])
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False)
            output.seek(0)

            return send_file(output, as_attachment=True, download_name="related_questions.xlsx",
                             mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

    return render_template('replied.html', related_questions=related_questions, distinct_states=distinct_states, related_count=len(related_questions))

@app.route('/add', methods=['POST'])
def add_question():
    memo_id = request.form['memo_id']
    state_name = request.form['state_name']
    question = request.form['new_question']
    reply = request.form['new_reply']

    if FAQ.query.filter_by(question=question, state_name=state_name).first():
        flash('Question already exists for this state!', 'warning')
    else:
        new_entry = FAQ(memo_id=memo_id, question=question, reply=reply, state_name=state_name)
        db.session.add(new_entry)
        db.session.commit()
        flash('Question added successfully!', 'success')
        logging.info(f'Question added manually: "{question}" with reply "{reply}" for state "{state_name}"')

    return redirect(url_for('replied'))

@app.route('/clear', methods=['POST'])
def clear():
    return redirect('/')

TEMPLATE_SECRET = "UPLOAD-VALID-123"

@app.route('/download-template')
def download_template():
    """Provide a downloadable Excel template with a hidden secret."""
    wb = Workbook()
    ws = wb.active
    ws.append(["question", "reply", "memo_id", "state_name"])

    wb.properties.comments = TEMPLATE_SECRET  # Hidden secret for validation
    wb.save("faq_template.xlsx")

    output = io.BytesIO()
    wb.save(output)
    output.seek(0)

    return send_file(output, as_attachment=True, download_name="faq_template.xlsx",
                     mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part', 'danger')
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            flash('No selected file', 'danger')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            if filename.endswith('.csv'):
                data = pd.read_csv(filepath)
            else:
                data = pd.read_excel(filepath)

            if 'question' in data.columns and 'reply' in data.columns:
                data = data.dropna(subset=['question', 'reply'])
                merged_count = 0
                duplicate_count = 0

                for _, row in data.iterrows():
                    question = row['question']
                    reply = row['reply']
                    memo_id = row.get('memo_id', '')
                    state_name = row.get('state_name', '')

                    if isinstance(question, str) and isinstance(reply, str):
                        if not FAQ.query.filter_by(question=question, state_name=state_name).first():
                            new_entry = FAQ(question=question, reply=reply, memo_id=memo_id, state_name=state_name)
                            db.session.add(new_entry)
                            merged_count += 1
                        else:
                            duplicate_count += 1

                db.session.commit()
                flash(f'{merged_count} questions merged and {duplicate_count} duplicates found!', 'success')
                logging.info(f'File {filename} processed: {merged_count} new questions added, {duplicate_count} duplicates')
            
            else:
                flash('Invalid file format! The file must contain "question" and "reply" columns.', 'danger')

            return redirect(url_for('replied'))

    return redirect(url_for('replied'))


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    #app.run(debug=True,host='0.0.0.0', port=int(os.getenv('PORT', 5000)))
    serve(app, host="0.0.0.0", port=int(os.getenv('PORT', 5000)))

