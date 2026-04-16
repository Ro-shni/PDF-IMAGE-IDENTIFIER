#working, but issue displaying images for urls rest is fine, creating a new one for unique header names
import requests
from io import BytesIO
from flask import Flask, request, render_template, send_from_directory
from flask import Flask, request, session, jsonify, render_template
import requests_oauthlib
import fitz  #PyMuPDF
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import yaml
from flask_mwoauth import MWOAuth
from urllib.parse import urlparse

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}

# Load configuration from YAML file
__dir__ = os.path.dirname(__file__)
app.config.update(yaml.safe_load(open(os.path.join(__dir__, 'config.yaml'))))

BASE_URL = app.config['OAUTH_MWURI']
CONSUMER_KEY = app.config['CONSUMER_KEY']
CONSUMER_SECRET = app.config['CONSUMER_SECRET']

# Register blueprint to app
MW_OAUTH = MWOAuth(
    base_url=BASE_URL,
    consumer_key=CONSUMER_KEY,
    consumer_secret=CONSUMER_SECRET
)
app.register_blueprint(MW_OAUTH.bp)

# Ensure upload directory exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Ensure static folder exists
if not os.path.exists('static'):
    os.makedirs('static')

def allowed_file(filename):
    """Check if the uploaded file is a PDF."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def authenticated_session():
    if 'mwoauth_access_token' in session:
        auth = requests_oauthlib.OAuth1(
            client_key=CONSUMER_KEY,
            client_secret=CONSUMER_SECRET,
            resource_owner_key=session['mwoauth_access_token']['key'],
            resource_owner_secret=session['mwoauth_access_token']['secret']
        )
        return auth

    return None

def get_authenticated_token():
    auth = authenticated_session()
    if auth:
        # The authenticated session is returned from `authenticated_session()`
        return auth
    else:
        print("Authentication failed or token not found.")
        return None

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

def extract_pdf_name(pdf_url):
    if isinstance(pdf_url, bytes):  # Check if the pdf_url is in bytes
        pdf_url = pdf_url.decode('utf-8')  # Decode to string if it's in bytes

    pdf_name = os.path.basename(pdf_url)  # Extract the name of the PDF
    pdf_name = pdf_name.replace(' ', '_').replace('-', '_')
    
    return pdf_name



static_uploads_dir = os.path.join('static', 'uploads')
if not os.path.exists(static_uploads_dir):
    os.makedirs(static_uploads_dir)

def display_extracted_images(cropped_images, page_num, pdf_url):
    """Display cropped image regions and save them under static/uploads folder."""
    # Extract the PDF name from the URL
    pdf_name = extract_pdf_name(pdf_url)
    
    cropped_image_paths = []
    for i, cropped in enumerate(cropped_images):
        # Use the extracted PDF name for the image filenames
        cropped_image_path = f"{pdf_name}_page{page_num+1}_image{i+1}.png"
        cropped_image_full_path = os.path.join(static_uploads_dir, cropped_image_path)
        
        # Ensure the static/uploads folder exists
        if not os.path.exists(static_uploads_dir):
            os.makedirs(static_uploads_dir)
        
        # Save the image using the updated filename
        plt.imsave(cropped_image_full_path, cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
        cropped_image_paths.append(f"{cropped_image_path}")  # Use relative path here
    
    return cropped_image_paths

def process_pdf_and_extract_images(pdf_data, pdf_url, zoom_x=2.0, zoom_y=2.0, min_contour_area=1500, image_area_threshold=20000):
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
            cropped_images_paths[page_num + 1] = display_extracted_images(cropped_images, page_num, pdf_url)
            
            # Save each cropped image for "Image" pages
            for i, cropped in enumerate(cropped_images):
                # Construct file name for the image
                file_name = f"{os.path.basename(pdf_url).replace('.pdf', '')}_page{page_num+1}_image{i+1}.png"
                file_path = os.path.join('static', 'uploads', file_name)
                
                # Ensure the directory exists
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                
                # Save image to static directory
                plt.imsave(file_path, cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))

        elif classification == "Image+Text":
            text_image_pages.append(page_num + 1)
            cropped_images = extract_large_contours(img, contours, min_contour_area=image_area_threshold)
            cropped_images_paths[page_num + 1] = display_extracted_images(cropped_images, page_num, pdf_url)

            # Save each cropped image for "Image+Text" pages
            for i, cropped in enumerate(cropped_images):
                # Construct file name for the image
                file_name = f"{os.path.basename(pdf_url).replace('.pdf', '')}_page{page_num+1}_image{i+1}.png"
                file_path = os.path.join('static', 'uploads', file_name)
                
                # Ensure the directory exists
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                
                # Save image to static directory
                plt.imsave(file_path, cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))

    return image_pages, text_image_pages, cropped_images_paths


def fetch_pdf_with_user_agent(pdf_url):
    """Fetch the PDF file using a custom User-Agent."""
    headers = {'User-Agent': 'wiki.py/1.0 (roshninekkanti@gmail.com) Python-requests'}
    response = requests.get(pdf_url, headers=headers,allow_redirects=True)
    if response.status_code == 200:
        return BytesIO(response.content)
    else:
        response.raise_for_status()


from requests_oauthlib import OAuth1
from datetime import datetime
import requests
import requests
import urllib.parse
import re
from datetime import datetime

def fetch_metadata(title):
    """Fetch metadata from Wikimedia Commons for a given file title."""
    url = "https://commons.wikimedia.org/w/api.php"

    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "imageinfo",
        "iiprop": "timestamp|user|comment|url|metadata",
    }

    response = requests.get(url, params=params, verify=True)
    data = response.json()

    pages = data.get("query", {}).get("pages", {})
    for page_id, page_data in pages.items():
        if "imageinfo" in page_data:
            info = page_data["imageinfo"][0]
            return {
                "author": info.get("user", "Unknown"),
                "timestamp": info.get("timestamp", ""),
                "comment": info.get("comment", ""),
                "source": info.get("url", ""),
            }

    return None  # Return None if metadata isn't found

def upload_to_commons(image_path, filename, auth_ses=None):
    """Uploads an image to Wikimedia Commons with correct metadata."""
    if auth_ses is not None:
        endpoint_url = "https://commons.wikimedia.org/w/api.php"

        # Step 1: Get CSRF token
        token_response = requests.get(endpoint_url, params={
            'action': 'query',
            'meta': 'tokens',
            'format': 'json'
        }, auth=auth_ses, verify=True)

        if token_response.status_code != 200:
            return f"Error retrieving token: {token_response.status_code} - {token_response.text}"

        token_data = token_response.json()
        csrftoken = token_data['query']['tokens']['csrftoken']

        # Step 2: Clean filename (fix illegal characters)
        cleaned_filename = urllib.parse.unquote(filename)
        cleaned_filename = re.sub(r'[^\w.-]', '_', cleaned_filename)

        # Step 3: Fetch existing metadata
        metadata = fetch_metadata(filename)
        if metadata:
            author = metadata["author"]
            source = metadata["source"]
            description = metadata["comment"] if metadata["comment"] else "Uploaded file"
        else:
            author = "Unknown"
            source = "Unknown"
            description = "Uploaded file"

        # Generate metadata text
        current_date = datetime.now().strftime('%Y-%m-%d')
        text_details = f"""=={{int:filedesc}}==
{{{{Information
|description={description}
|date={current_date}
|source={source}
|author=[[User:{author}|{author}]]
}}}}
=={{int:license-header}}==
{{{{self|cc-by-sa-4.0}}}}
[[Category:Uploaded PDFs]]
{{{{Extracted from|File:{cleaned_filename}}}}}
"""

        # Step 4: Upload file
        upload_params = {
            'action': 'upload',
            'filename': cleaned_filename,
            "format": "json",
            "token": csrftoken,
            "text": text_details,  # Attach correct metadata
            "ignorewarnings": 1,
        }

        file = {'file': open(image_path, 'rb')}
        upload_response = requests.post(endpoint_url, data=upload_params, files=file, auth=auth_ses, verify=True)

        if upload_response.status_code != 200:
            return f"Error uploading file: {upload_response.status_code} - {upload_response.text}"

        return upload_response.json()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('file')
        url = request.form.get('url')

        if file and allowed_file(file.filename):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            image_pages, text_image_pages, cropped_images_paths = process_pdf_and_extract_images(file_path, pdf_url=file_path)

            return render_template('review.html', image_pages=image_pages, text_image_pages=text_image_pages, cropped_images_paths=cropped_images_paths)

        elif url:
            try:
                pdf_data = fetch_pdf_with_user_agent(url)
                image_pages, text_image_pages, cropped_images_paths = process_pdf_and_extract_images(pdf_data, pdf_url=url)

                return render_template('review.html', image_pages=image_pages, text_image_pages=text_image_pages, cropped_images_paths=cropped_images_paths)

            except requests.exceptions.RequestException as e:
                return f"Error fetching the PDF: {e}", 400

    return render_template('index.html')

@app.route('/upload_selected_images', methods=['POST'])
def upload_selected_images():
    selected_images = request.form.getlist('selected_images')
    if not selected_images:
        return "No images selected", 400

    auth_ses = get_authenticated_token()
    if not auth_ses:
        return "Authentication failed", 400

    upload_results = []
    base_dir = os.path.join(app.root_path, 'static/uploads')
    for image_path in selected_images:
        # Ensure correct absolute path
        absolute_image_path = os.path.join(base_dir, os.path.basename(image_path))
        filename = os.path.basename(absolute_image_path)

        # Check if the file exists before uploading
        if not os.path.exists(absolute_image_path):
            return f"File not found: {absolute_image_path}", 404

        result = upload_to_commons(absolute_image_path, filename, auth_ses)
        upload_results.append(result)

    return render_template('upload_results.html', upload_results=upload_results)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

def getUser():
    if MW_OAUTH.get_current_user(True) is not None:
        return MW_OAUTH.get_current_user(True)
    else:
        return None

@app.route('/upload_metadata', methods=['POST'])
def upload_metadata():
    # Receive the title from the request's JSON payload
    data = request.get_json()
    title = data.get('title', '')  # Assuming 'title' is passed in the request
    # Check if the title is valid
    if not title:
        return jsonify({"error": "No title provided"}), 400
    # Fetch metadata using the fetch_metadata function
    metadata = fetch_metadata(title)
    
    if metadata:
        return jsonify({"message": "Metadata fetched successfully", "metadata": metadata}), 200
    else:
        return jsonify({"error": "No PDFs found for the given title"}), 404

    return jsonify({"metadata": metadata}), 200
if __name__ == '__main__':
    app.run(debug=True)