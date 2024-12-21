from flask import Flask, request, render_template, send_file, jsonify
import os
import requests
from werkzeug.utils import secure_filename
from io import BytesIO
import fitz  # PyMuPDF
import cv2
import numpy as np
import shutil

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
CROPPED_FOLDER = 'cropped'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CROPPED_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {'pdf'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_pdf_as_images(pdf_data, zoom_x=2.0, zoom_y=2.0):
    """Load a PDF file and convert each page into an image."""
    if isinstance(pdf_data, BytesIO):
        pdf_data.seek(0)  # Reset pointer to the start
        doc = fitz.open(stream=pdf_data.read(), filetype="pdf")
    else:
        doc = fitz.open(pdf_data)
    
    images = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        mat = fitz.Matrix(zoom_x, zoom_y)
        pix = page.get_pixmap(matrix=mat)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 4:  # Convert RGBA to RGB if needed
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        images.append(img)
    doc.close()
    return images

def preprocess_image(img, kernel_size=(5, 5), sigma=10):
    """Apply Gaussian blur to reduce noise."""
    blurred = cv2.GaussianBlur(img, kernel_size, sigma)
    return blurred

def threshold_and_find_contours(blurred_img):
    """Threshold the image and find contours."""
    _, binary = cv2.threshold(blurred_img, 225, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours, binary

def classify_image(contours, img_shape, min_contour_area=1500, image_area_threshold=20000):
    """Classify the page as 'Image', 'Image+Text', or 'Text'."""
    page_area = img_shape[0] * img_shape[1]

    if not contours:
        return "Text"  # No contours found, assume text

    filtered_contours = [c for c in contours if cv2.contourArea(c) > min_contour_area]

    if not filtered_contours:
        return "Text"  # No significant contours, assume text

    total_contour_area = sum(cv2.contourArea(c) for c in filtered_contours)
    contour_area_ratio = total_contour_area / page_area

    large_contours = [c for c in filtered_contours if cv2.contourArea(c) > image_area_threshold]

    if large_contours:
        return "Image" if len(filtered_contours) == len(large_contours) else "Image+Text"

    return "Text"

def extract_large_contours(img, contours, min_contour_area=20000):
    """Extract regions of large contours."""
    large_contours = [c for c in contours if cv2.contourArea(c) > min_contour_area]
    cropped_images = []
    for contour in large_contours:
        x, y, w, h = cv2.boundingRect(contour)
        cropped = img[y:y+h, x:x+w]
        cropped_images.append(cropped)
    return cropped_images

def process_pdf(file_path):
    images = load_pdf_as_images(file_path)
    cropped_paths = []

    for idx, img in enumerate(images):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = preprocess_image(gray)
        contours, _ = threshold_and_find_contours(blurred)
        cropped_images = extract_large_contours(img, contours)

        for i, cropped in enumerate(cropped_images):
            cropped_path = os.path.join(CROPPED_FOLDER, f'cropped_{idx}_{i}.png')
            cv2.imwrite(cropped_path, cropped)
            cropped_paths.append(cropped_path)

    return cropped_paths

def extract_pdf_name(pdf_url):
    if isinstance(pdf_url, bytes):  # Check if the pdf_url is in bytes
        pdf_url = pdf_url.decode('utf-8')  # Decode to string if it's in bytes

    pdf_name = os.path.basename(pdf_url)  # Extract the name of the PDF
    pdf_name = pdf_name.replace(' ', '_').replace('-', '_')
    
    return pdf_name

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files and 'url' not in request.form:
            return "No file or URL provided", 400

        if 'file' in request.files:
            file = request.files['file']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(UPLOAD_FOLDER, filename)
                file.save(file_path)

        elif 'url' in request.form:
            pdf_url = request.form['url']
            response = requests.get(pdf_url)

            if response.status_code == 200:
                filename = extract_pdf_name(pdf_url)
                file_path = os.path.join(UPLOAD_FOLDER, filename)
                with open(file_path, 'wb') as f:
                    f.write(response.content)
            else:
                return "Failed to download the PDF from the provided URL", 400

        else:
            return "Invalid input", 400

        cropped_paths = process_pdf(file_path)
        return jsonify(cropped_paths=cropped_paths)

    return render_template('upload.html')

@app.route('/download/<filename>')
def download_file(filename):
    file_path = os.path.join(CROPPED_FOLDER, filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    return "File not found", 404

@app.route('/cleanup', methods=['POST'])
def cleanup():
    """Endpoint to clean up uploaded and cropped files."""
    try:
        shutil.rmtree(UPLOAD_FOLDER)
        shutil.rmtree(CROPPED_FOLDER)
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(CROPPED_FOLDER, exist_ok=True)
        return "Cleanup successful", 200
    except Exception as e:
        return str(e), 500

if __name__ == '__main__':
    app.run(debug=True)
