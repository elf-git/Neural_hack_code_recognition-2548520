import cv2
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="Postal Code Digit Recognition", page_icon="📮", layout="wide")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model('mnist_model.h5')

def segment_and_predict(image_array, model):
    """
    Takes a grayscale numpy image, extracts individual digits using contours,
    sorts them left-to-right, predicts each digit using the CNN,
    and returns the predicted string along with the bounding boxes.
    """
    # Threshold the image
    _, thresh = cv2.threshold(image_array, 10, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours based on size to remove noise
    digit_contours = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > 5 and h > 15: # Arbitrary small thresholds to avoid tiny specks
            digit_contours.append((x, y, w, h))
            
    # Sort contours from left to right
    digit_contours = sorted(digit_contours, key=lambda b: b[0])
    
    predictions = []
    processed_rois = []
    
    for x, y, w, h in digit_contours:
        # Extract the region of interest (ROI)
        # Adding some padding around the bounding box
        pad = min(w, h) // 4
        x_start = max(0, x - pad)
        y_start = max(0, y - pad)
        x_end = min(image_array.shape[1], x + w + pad)
        y_end = min(image_array.shape[0], y + h + pad)
        
        roi = image_array[y_start:y_end, x_start:x_end]
        
        if roi.size == 0:
            continue
            
        # Resize to 28x28
        roi_resized = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
        processed_rois.append(roi_resized)
        
        # Normalize and reshape for prediction
        roi_norm = roi_resized / 255.0
        roi_input = roi_norm.reshape(1, 28, 28, 1)
        
        # Predict
        pred_probs = model.predict(roi_input)
        pred_class = np.argmax(pred_probs[0])
        predictions.append(str(pred_class))
        
    return "".join(predictions), digit_contours, processed_rois

# Main layout
st.title("📮 Postal Code Digit Recognition")
st.markdown("Draw a multi-digit postal code, or upload an image containing a handwritten code. The CNN will process and predict all digits from left to right!")

try:
    model = load_model()
except Exception as e:
    st.error("Model not found. Please train the model first by running `python train.py`.")
    st.stop()
    
# Layout Modes
mode = st.radio("Choose Input Method:", ["Interactive Canvas", "Image Upload"])

if mode == "Interactive Canvas":
    st.subheader("Draw Postal Code Here")
    # Add a wider interactive canvas
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0.3)",
        stroke_width=15, 
        stroke_color="#FFFFFF", 
        background_color="#000000", 
        update_streamlit=True,
        height=200,
        width=500, # Wider to accommodate postal codes
        drawing_mode="freedraw",
        key="canvas",
    )
    
    if st.button("Reset / Clear"):
        st.rerun()

    st.subheader("Prediction")
    
    if canvas_result.image_data is not None:
        img_array = canvas_result.image_data
        if np.any(img_array[:, :, 3] > 0):
            # Convert to grayscale
            gray_img = np.mean(img_array[:, :, :3], axis=2).astype(np.uint8)
            
            # Predict
            prediction, bboxes, rois = segment_and_predict(gray_img, model)
            
            if prediction:
                st.success(f"### Predicted Postal Code: **{prediction}**")
                
                # Draw bounding boxes on the original canvas image
                color_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)
                for x, y, w, h in bboxes:
                    cv2.rectangle(color_img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    
                st.image(color_img, caption="Processed Full Image with Detected Digits", width=500)
                
                st.markdown("**Segmented Digits Extracted:**")
                cols = st.columns(len(rois) if rois else 1)
                for i, roi in enumerate(rois):
                    # Display the individual 28x28 segmented rois being sent to model
                    cols[i].image(roi, caption=f"Digit {i+1}: '{prediction[i]}'", width=80)
            else:
                st.info("No recognizable digits found.")
        else:
            st.info("Please draw a postal code on the canvas.")

elif mode == "Image Upload":
    uploaded_file = st.file_uploader("Upload an image of a handwritten postal code", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        col1, col2 = st.columns([1, 1])
        
        image = Image.open(uploaded_file)
        # Convert to numpy and grayscale
        img_array = np.array(image.convert("L"))
        
        # Invert colors if necessary Assuming training was white-on-black (MNIST)
        # We need the user image to be white digit on black background
        # Simple heuristic: if corners are bright, the background is probably light
        if img_array[0,0] > 127:
            img_array = 255 - img_array
        
        with col1:
            st.subheader("Uploaded Image")
            st.image(image, use_column_width=True)
            
        with col2:
            st.subheader("Prediction")
            prediction, bboxes, rois = segment_and_predict(img_array, model)
            
            if prediction:
                st.success(f"### Predicted Postal Code: **{prediction}**")
                
                # Draw bounding boxes on image for visualization
                color_img = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
                for x, y, w, h in bboxes:
                    cv2.rectangle(color_img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    
                st.image(color_img, caption="Processed Full Image with Detected Digits", use_column_width=True)
            else:
                st.warning("Could not segment any digits from the image. Ensure the image is clear.")

st.markdown("---")
st.markdown("### How it works")
st.markdown("1. **Segmentation**: Uses OpenCV Contour Detection to separately slice each individual handwritten digit.")
st.markdown("2. **Preprocessing**: Extracted digits are ordered from left to right, cropped, padded, and resized to $28 \\times 28$ pixels.")
st.markdown("3. **Classification**: The CNN evaluates each $28 \\times 28$ patch independently and stitches the text string back as a complete postal code.")
