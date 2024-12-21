import requests
from io import BytesIO
from flask import Flask, request, render_template, send_from_directory
import fitz  # PyMuPDF
import cv2
import numpy as np
import os
import statistics

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}

# Ensure upload directory exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

def allowed_file(filename):
    """Check if the uploaded file is a PDF."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def load_pdf_as_images(pdf_data, zoom_x=2.0, zoom_y=2.0):
    """Load a PDF file and convert each page into an image."""
    doc = fitz.open(pdf_data)
    images = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        mat = fitz.Matrix(zoom_x, zoom_y)
        pix = page.get_pixmap(matrix=mat)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 4:  # Remove alpha channel if present
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        images.append(img)
    doc.close()
    return images

def classify_image(contours, img_shape, min_contour_area=1500, image_area_threshold=20000):
    """Classify the page based on contour analysis."""
    page_area = img_shape[0] * img_shape[1]

    if not contours:
        return "Text"

    filtered_contours = [c for c in contours if cv2.contourArea(c) > min_contour_area]

    if not filtered_contours:
        return "Text"

    mean_area = statistics.mean([cv2.contourArea(c) for c in filtered_contours])
    total_contour_area = sum([cv2.contourArea(c) for c in filtered_contours])
    contour_area_ratio = total_contour_area / page_area

    large_contours = [c for c in contours if cv2.contourArea(c) > image_area_threshold]

    if large_contours and filtered_contours:
        return "Image+Text"
    if mean_area > image_area_threshold or contour_area_ratio > 0.5:
        return "Image"
    return "Text"

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files.get('file')
        url = request.form.get('url')  # Assuming you have a form field for URL

        if file and allowed_file(file.filename):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            # Process the uploaded PDF
            images = load_pdf_as_images(file_path)
            image_pages = []
            text_image_pages = []

            for page_num, img in enumerate(images):
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                blurred = cv2.GaussianBlur(gray, (5, 5), 10)
                _, binary = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY_INV)
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                classification = classify_image(contours, img.shape)
                if classification == "Image":
                    image_pages.append(page_num + 1)
                elif classification == "Image+Text":
                    text_image_pages.append(page_num + 1)

            return render_template('results.html', image_pages=image_pages, text_image_pages=text_image_pages)

        elif url:
            # Process direct PDF URL
            try:
                response = requests.get(url)
                response.raise_for_status()
                pdf_data = BytesIO(response.content)

                # Process the PDF from the URL
                images = load_pdf_as_images(pdf_data)
                image_pages = []
                text_image_pages = []

                for page_num, img in enumerate(images):
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    blurred = cv2.GaussianBlur(gray, (5, 5), 10)
                    _, binary = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY_INV)
                    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    classification = classify_image(contours, img.shape)
                    if classification == "Image":
                        image_pages.append(page_num + 1)
                    elif classification == "Image+Text":
                        text_image_pages.append(page_num + 1)

                return render_template('results.html', image_pages=image_pages, text_image_pages=text_image_pages)
            except requests.exceptions.RequestException as e:
                return f"Error fetching the PDF: {e}", 400

    return render_template('upload.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
