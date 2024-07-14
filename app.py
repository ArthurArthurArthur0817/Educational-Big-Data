import os
import subprocess
from flask import Flask, request, render_template, send_file, send_from_directory

app = Flask(__name__)

# 文件上传目录
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    if 'pdf_files' not in request.files or 'question_file' not in request.files:
        return "No file part", 400

    pdf_files = request.files.getlist('pdf_files')
    question_file = request.files['question_file']

    pdf_paths = []
    for pdf_file in pdf_files:
        pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], pdf_file.filename)
        pdf_file.save(pdf_path)
        pdf_paths.append(os.path.abspath(pdf_path))

    question_path = os.path.join(app.config['UPLOAD_FOLDER'], question_file.filename)
    question_file.save(question_path)

    # 确认文件路径
    print("PDF Paths:", pdf_paths)
    print("Question Path:", os.path.abspath(question_path))

    # 调用 mistral-7B-instruct.py
    mistral_command = ["python", "mistral-7B-instruct.py", os.path.abspath(question_path)] + pdf_paths
    print("Running command:", mistral_command)  # 打印命令用于调试
    mistral_result = subprocess.run(mistral_command, capture_output=True, text=True)
    print("Mistral output:", mistral_result.stdout)
    print("Mistral error:", mistral_result.stderr)

    # 调用 gemini.py
    gemini_command = ["python", "gemini.py"]
    print("Running command:", gemini_command)  # 打印命令用于调试
    gemini_result = subprocess.run(gemini_command, capture_output=True, text=True)
    print("Gemini output:", gemini_result.stdout)
    print("Gemini error:", gemini_result.stderr)

    return render_template('results.html')

@app.route('/download')
def download_file():
    path = "gemini_responses.txt"
    return send_file(path, as_attachment=True)

@app.route('/hammett')
def hammett():
    return render_template('hammett_plot.html')

@app.route('/plot', methods=['POST'])
def plot():
    from hammett_plot import generate_hammett_plot
    y_axis = request.form['y_axis']
    plot_file = request.files['plot_file']
    plot_path = os.path.join(app.config['UPLOAD_FOLDER'], plot_file.filename)
    plot_file.save(plot_path)

    image_path, output_path = generate_hammett_plot(y_axis, plot_path, app.config['UPLOAD_FOLDER'])

    return render_template('plot_result.html', image_path=image_path, output_path=output_path)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
