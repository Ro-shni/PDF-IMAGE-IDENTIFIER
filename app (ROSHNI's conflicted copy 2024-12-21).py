import os
import fitz  # PyMuPDF
import cv2
import numpy as np
import statistics
import streamlit as st

# Function to save uploaded file temporarily
def save_uploaded_file(uploaded_file, save_dir="temp"):
    os.makedirs(save_dir, exist_ok=True)  # Ensure the temp directory exists
    file_path = os.path.join(save_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())
    return file_path

# Function to load PDF and process as images
def load_pdf_as_images(pdf_path, zoom_x=2.0, zoom_y=2.0):
    doc = fitz.open(pdf_path)
    images = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)  # Load page
        mat = fitz.Matrix(zoom_x, zoom_y)  # Create a transformation matrix for zoom
        pix = page.get_pixmap(matrix=mat)  # Render page as an image with the zoom
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 4:  # If the image has an alpha channel, remove it
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        images.append(img)
    doc.close()
    return images

# Preprocessing and other functions remain the same
def preprocess_image(img, kernel_size=(5, 5), sigma=10):
    blurred = cv2.GaussianBlur(img, kernel_size, sigma)
    return blurred

def threshold_and_find_contours(blurred_img):
    _, binary = cv2.threshold(blurred_img, 225, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours, binary

def classify_image(contours, img_shape, min_contour_area=1500, image_area_threshold=20000):
    page_area = img_shape[0] * img_shape[1]

    if not contours:
        return "Text"

    filtered_contours = [c for c in contours if cv2.contourArea(c) > min_contour_area]
    if not filtered_contours:
        return "Text"

    mean_area = statistics.mean([cv2.contourArea(c) for c in filtered_contours])
    total_contour_area = sum([cv2.contourArea(c) for c in filtered_contours])
    contour_area_ratio = total_contour_area / page_area

    small_contours = [c for c in contours if cv2.contourArea(c) <= min_contour_area]
    large_contours = [c for c in contours if cv2.contourArea(c) > image_area_threshold]

    if large_contours and small_contours:
        return "Image+Text"

    if mean_area > image_area_threshold or contour_area_ratio > 0.5:
        return "Image"

    return "Text"

def process_pdf_pages(pdf_path, zoom_x=2.0, zoom_y=2.0, min_contour_area=1500, image_area_threshold=20000):
    image_pages = []
    text_image_pages = []

    images = load_pdf_as_images(pdf_path, zoom_x=zoom_x, zoom_y=zoom_y)
    for page_num, img in enumerate(images):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred_img = preprocess_image(gray, kernel_size=(5, 5), sigma=10)
        contours, _ = threshold_and_find_contours(blurred_img)
        classification = classify_image(contours, img.shape, min_contour_area=min_contour_area, image_area_threshold=image_area_threshold)

        if classification == "Image":
            image_pages.append(page_num + 1)
        elif classification == "Image+Text":
            text_image_pages.append(page_num + 1)

    return image_pages, text_image_pages

# Streamlit Interface
st.title("PDF Page Classifier")
st.write("Upload a PDF to classify pages as text, image, or both.")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    with st.spinner("Processing..."):
        # Save file and get the path
        pdf_path = save_uploaded_file(uploaded_file)

        # Process the file
        image_pages, text_image_pages = process_pdf_pages(pdf_path, zoom_x=2.0, zoom_y=2.0)

        # Cleanup: Remove the file after processing
        os.remove(pdf_path)

    # Display Results
    if image_pages:
        st.write(f"Pages with only images: {image_pages}")
    else:
        st.write("No pages with only images found.")

    if text_image_pages:
        st.write(f"Pages with images and text: {text_image_pages}")
    else:
        st.write("No pages with images and text found.")
