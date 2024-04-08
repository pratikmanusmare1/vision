import streamlit as st
from streamlit.components.v1 import html
import tempfile
import os
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
import openai
import cv2
import numpy as np

openai.api_key = "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"

def generate_response(prompt):
    """
    Function to generate response based on prompt.

    Parameters:
    - prompt: the text prompt for generating the response.

    Returns:
        - The generated response
    """
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "AI and human interaction."},
                  {"role": "user", "content": prompt}],
    )
    text_response = response['choices'][0]['message']['content'].strip()
    return text_response

def analyze_image(image_path):
    """
    Function to perform image analysis using a pre-trained model.

    Parameters:
    - image_path: The path to the image file to be analyzed.
        
    Returns:
        - The top prediction label and probabilities
    """
    try:
        img = Image.open(image_path).convert('RGB')
    except:
        st.error("Error: Invalid image file. Please upload a valid image file.")
        return None
    
    img_resized = img.resize((224, 224))
    img_array = preprocess_input(tf.keras.preprocessing.image.img_to_array(img_resized))
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    model = MobileNetV2(weights='imagenet')
    predictions = model.predict(img_array)
    results = decode_predictions(predictions, top=5)[0]

    labels = [result[1] for result in results]
    probabilities = [result[2] for result in results]

    return labels[0], probabilities

def ask_questions_and_display_answers(food_item, probabilities):
    """
    Function to generate and display the food item and probabilities without user input.

    Parameters:
    - food_item: The recognized food item from image analysis.
    - probabilities: The probabilities associated with the recognized food item.
    """
    st.write("<div class='food-item'>Food Item: {}</div>".format(food_item), unsafe_allow_html=True)
    answers = {
        "Protein per 100 gram": round(probabilities[0] * 100, 2),
        "Calories in 100 gram": round(probabilities[1] * 100, 2),
        "Sugar in 100 gram": round(probabilities[2] * 100, 2),
        "Fat per 100 gram": round(probabilities[3] * 100, 2),
        "Carbs per 100 gram": round(probabilities[4] * 100, 2)
    }
    for question, answer in answers.items():
        st.write("<div class='probability-label'>{}</div><div class='probability-value'>\"{}\"</div>".format(question, answer), unsafe_allow_html=True)

def capture_image_from_webcam():
    """
    Function to capture an image from the webcam.
    """
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)  # Width
    cap.set(4, 480)  # Height
    cap.set(10, 100)  # Brightness

    while True:
        success, img = cap.read()
        cv2.imshow("Webcam", img)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            cv2.imwrite("captured_image.jpg", img)
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    """
    Main function for running the Streamlit web application.
    """
    st.title("VISION GPT")

    # Options for image upload or webcam capture
    option = st.radio("Select an option:", ("Upload Image", "Open Webcam"))

    if option == "Upload Image":
        img = st.file_uploader("")

        if img:
            temp_dir = tempfile.mkdtemp()
            img_path = os.path.join(temp_dir, img.name)
            with open(img_path, "wb") as f:
                f.write(img.getvalue())

            st.image(img, caption='Uploaded Image', use_column_width=True)

            with st.spinner('Analyzing image...'):
                food_item, probabilities = analyze_image(img_path)
                if food_item:

                    # Ask questions and display answers
                    ask_questions_and_display_answers(food_item, probabilities)

    elif option == "Open Webcam":
        st.write("Press 's' to capture image from the webcam.")
        if st.button("Open Webcam"):
            capture_image_from_webcam()
            st.write("Image captured successfully!")
            img_path = "captured_image.jpg"
            img = Image.open(img_path)
            st.image(img, caption='Captured Image', use_column_width=True)

            with st.spinner('Analyzing image...'):
                food_item, probabilities = analyze_image(img_path)
                if food_item:
                    # Ask questions and display answers
                    ask_questions_and_display_answers(food_item, probabilities)

if __name__ == "__main__":
    main()
