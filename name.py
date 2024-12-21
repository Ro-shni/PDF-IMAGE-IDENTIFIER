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
        cropped_image_paths.append(f"/static/uploads/{cropped_image_path}")  # Use relative path here
    
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
            
            # After processing, upload each image to Wikimedia Commons
            for i, cropped in enumerate(cropped_images):
                # Construct file name for the image
                file_name = f"{pdf_url.split('/')[-1]}_page{page_num+1}_image{i+1}.png"
                file_path = os.path.join('static', file_name)
                
                # Save image to static directory first
                plt.imsave(file_path, cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
                
                # Now upload to Wikimedia Commons
                auth_ses = get_authenticated_token()  # Get the OAuth token using the authenticated session
                if auth_ses:
                    upload_to_commons(file_path, file_name, auth_ses)

        elif classification == "Image+Text":
            text_image_pages.append(page_num + 1)
            cropped_images = extract_large_contours(img, contours, min_contour_area=image_area_threshold)
            cropped_images_paths[page_num + 1] = display_extracted_images(cropped_images, page_num, pdf_url)

    return image_pages, text_image_pages, cropped_images_paths



def fetch_pdf_with_user_agent(pdf_url):
    """Fetch the PDF file using a custom User-Agent."""
    headers = {'User-Agent': 'wiki.py/1.0 (roshninekkanti@gmail.com) Python-requests'}
    response = requests.get(pdf_url, headers=headers)
    if response.status_code == 200:

        return BytesIO(response.content)
    else:
        response.raise_for_status()


from requests_oauthlib import OAuth1

def upload_to_commons(image_path, filename, auth_ses=None):
    """Uploads an image to Wikimedia Commons."""
    # Step 1: Get the upload token if not provided
    if auth_ses is not None:
        endpoint_url = "https://commons.wikimedia.org/w/api.php"
        crsf_params = {
            'action': 'query',
            'meta': 'tokens',
            'format': 'json'
        }
        token_response = requests.get(endpoint_url, params=crsf_params, auth=auth_ses)

        # Check if the response is valid
        if token_response.status_code != 200:
            return f"Error retrieving token: {token_response.status_code} - {token_response.text}"

        try:
            token_data = token_response.json()
        except requests.exceptions.JSONDecodeError:
            return f"Failed to decode JSON response: {token_response.text}"

        # Extract the upload token from the response
        csrftoken = token_data['query']['tokens']['csrftoken']

        print(f"Retrieved CSRF token: {csrftoken}")


        # Step 2: Prepare the file for upload
        upload_params = {
            'action': 'upload',
            'filename': filename,
            'token': csrftoken,
            'description': 'Uploaded via API', 
            'source': 'Own work', 
        }

        # Step 3: Perform the upload request with the OAuth authentication
        # Read the file for POST request
        file = {
            'file': open(image_path, 'rb')
        }
        upload_response = requests.post(endpoint_url, data=upload_params, files=file, auth=auth_ses)

        # Check if the upload request was successful
        if upload_response.status_code != 200:
            return f"Error uploading file: {upload_response.status_code} - {upload_response.text}"

        # Try to parse the JSON response
        try:
            response_json = upload_response.json()
            # Check if the response contains a success message
            print(response_json)
        except requests.exceptions.JSONDecodeError:
            return f"Failed to decode JSON response: {upload_response.text}"



@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('file')
        url = request.form.get('url')  # Assuming you have a form field for URL

        if file and allowed_file(file.filename):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            # Process the uploaded PDF from the file path
            image_pages, text_image_pages, cropped_images_paths = process_pdf_and_extract_images(file_path, pdf_url=file_path)

            return render_template('r1.html', image_pages=image_pages, text_image_pages=text_image_pages, cropped_images_paths=cropped_images_paths)

        elif url:
            try:
                pdf_data = fetch_pdf_with_user_agent(url)

                # Process the PDF from the URL
                image_pages, text_image_pages, cropped_images_paths = process_pdf_and_extract_images(pdf_data, pdf_url=url)

                return render_template('r1.html', image_pages=image_pages, text_image_pages=text_image_pages, cropped_images_paths=cropped_images_paths)

            except requests.exceptions.RequestException as e:
                return f"Error fetching the PDF: {e}", 400

    return render_template('u1.html', user=getUser())

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


def getUser():
    if MW_OAUTH.get_current_user(True) is not None:
        return MW_OAUTH.get_current_user(True)
    else:
        return None

if __name__ == '__main__':
    app.run(debug=True)