from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename
import joblib
from docx import Document

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'docx', 'txt'}

model = joblib.load('resume_classifier_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

def extract_text_from_docx(file_path):
    doc = Document(file_path)
    text = ''
    for para in doc.paragraphs:
        text += para.text + '\n'
    return text

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET', 'POST'])
def index():
    category = None
    if request.method == 'POST':
        if 'resume' not in request.files:
            return "No file part", 400

        file = request.files['resume']
        if file.filename == '':
            return "No selected file", 400

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            resume_text = extract_text_from_docx(file_path)
            X = vectorizer.transform([resume_text])
            category = model.predict(X)[0]

    return render_template('index.html', category=category)

if __name__ == '__main__':
    app.run(debug=True)
