#working, but issue displaying images for urls rest is fine, creating a new one for unique header names which is name.py
import requests
from io import BytesIO
from flask import Flask, request, render_template, send_from_directory
import fitz  # PyMuPDF
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}

# Ensure upload directory exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Ensure static folder exists
if not os.path.exists('static'):
    os.makedirs('static')

def allowed_file(filename):
    """Check if the uploaded file is a PDF."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

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

def display_extracted_images(cropped_images, page_num):
    """Display cropped image regions and save them under static folder."""
    cropped_image_paths = []
    for i, cropped in enumerate(cropped_images):
        # Save images in the 'static' folder, without 'static/' in the filename
        cropped_image_path = f"cropped_page{page_num+1}_image{i+1}.png"
        cropped_image_full_path = os.path.join('static', cropped_image_path)
        plt.imsave(cropped_image_full_path, cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
        cropped_image_paths.append(cropped_image_path)
    return cropped_image_paths


def process_pdf_and_extract_images(pdf_data, zoom_x=2.0, zoom_y=2.0, min_contour_area=1500, image_area_threshold=20000):
    """Process the PDF, classify pages, and extract large image regions."""
    images = load_pdf_as_images(pdf_data, zoom_x=zoom_x, zoom_y=zoom_y)
    image_pages = []
    text_image_pages = []
    cropped_images_paths = {}

    for page_num, img in enumerate(images):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred_img = preprocess_image(gray, kernel_size=(5, 5), sigma=10)
        contours, _ = threshold_and_find_contours(blurred_img)
        classification = classify_image(contours, img.shape, min_contour_area=min_contour_area, image_area_threshold=image_area_threshold)

        if classification == "Image":
            image_pages.append(page_num + 1)
            cropped_images = extract_large_contours(img, contours, min_contour_area=image_area_threshold)
            cropped_images_paths[page_num + 1] = display_extracted_images(cropped_images, page_num)
        elif classification == "Image+Text":
            text_image_pages.append(page_num + 1)
            cropped_images = extract_large_contours(img, contours, min_contour_area=image_area_threshold)
            cropped_images_paths[page_num + 1] = display_extracted_images(cropped_images, page_num)

    return image_pages, text_image_pages, cropped_images_paths

def fetch_pdf_with_user_agent(pdf_url):
    """Fetch the PDF file using a custom User-Agent."""
    headers = {'User-Agent': 'wiki.py/1.0 (roshninekkanti@gmail.com) Python-requests'}
    response = requests.get(pdf_url, headers=headers)
    if response.status_code == 200:

        return BytesIO(response.content)
    else:
        response.raise_for_status()

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files.get('file')
        url = request.form.get('url')  # Assuming you have a form field for URL

        if file and allowed_file(file.filename):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            # Process the uploaded PDF
            image_pages, text_image_pages, cropped_images_paths = process_pdf_and_extract_images(file_path)

            return render_template('r1.html', image_pages=image_pages, text_image_pages=text_image_pages, cropped_images_paths=cropped_images_paths)

        elif url:
            try:
                pdf_data = fetch_pdf_with_user_agent(url)

                # Process the PDF from the URL
                image_pages, text_image_pages, cropped_images_paths = process_pdf_and_extract_images(pdf_data)

                return render_template('r1.html', image_pages=image_pages, text_image_pages=text_image_pages, cropped_images_paths=cropped_images_paths)

            except requests.exceptions.RequestException as e:
                return f"Error fetching the PDF: {e}", 400

    return render_template('u1.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)

