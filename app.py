import streamlit as st
import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt

# Load the trained model
model = tf.keras.models.load_model("dogclassification.h5")

# Define class names
class_names = {
    "0": "Afghan", "1": "African Wild Dog", "2": "Airedale", "3": "American Hairless", "4": "American Spaniel",
    "5": "Basenji", "6": "Basset", "7": "Beagle", "8": "Bearded Collie", "9": "Bermaise", "10": "Bichon Frise",
    "11": "Blenheim", "12": "Bloodhound", "13": "Bluetick", "14": "Border Collie", "15": "Borzoi", "16": "Boston Terrier",
    "17": "Boxer", "18": "Bull Mastiff", "19": "Bull Terrier", "20": "Bulldog", "21": "Cairn", "22": "Chihuahua",
    "23": "Chinese Crested", "24": "Chow", "25": "Clumber", "26": "Cockapoo", "27": "Cocker", "28": "Collie",
    "29": "Corgi", "30": "Coyote", "31": "Dalmation", "32": "Dhole", "33": "Dingo", "34": "Doberman", "35": "Elk Hound",
    "36": "French Bulldog", "37": "German Sheperd", "38": "Golden Retriever", "39": "Great Dane", "40": "Great Perenees",
    "41": "Greyhound", "42": "Groenendael", "43": "Irish Spaniel", "44": "Irish Wolfhound", "45": "Japanese Spaniel",
    "46": "Komondor", "47": "Labradoodle", "48": "Labrador", "49": "Lhasa", "50": "Malinois", "51": "Maltese",
    "52": "Mex Hairless", "53": "Newfoundland", "54": "Pekinese", "55": "Pit Bull", "56": "Pomeranian", "57": "Poodle",
    "58": "Pug", "59": "Rhodesian", "60": "Rottweiler", "61": "Saint Bernard", "62": "Schnauzer", "63": "Scotch Terrier",
    "64": "Shar_Pei", "65": "Shiba Inu", "66": "Shih-Tzu", "67": "Siberian Husky", "68": "Vizsla", "69": "Yorkie"
}

# Function to get model accuracy
def get_model_accuracy():
    return "85%"

# Streamlit app
st.set_page_config(page_title="Dog Breed Classifier ", page_icon="🐶")

# Initialize session state for tracking page navigation and other data
if 'selection' not in st.session_state:
    st.session_state.selection = "Upload Image"

if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None

if 'processed_image' not in st.session_state:
    st.session_state.processed_image = None

if 'probabilities' not in st.session_state:
    st.session_state.probabilities = None

# Sidebar Navigation
with st.sidebar:
    st.title("Navigation")
    selection = st.radio("Go to", ["Upload Image", "Model Accuracy", "Graphs"], index=["Upload Image", "Model Accuracy", "Graphs"].index(st.session_state.selection))

    st.markdown(
        """
        <style>
        .sidebar {
            background-color: #7fff00;
            color: blue;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if selection == "Model Accuracy":
        st.write(f"Model Accuracy: {get_model_accuracy()}")

# Main content
if selection == "Upload Image":
    st.markdown(
        """
        <style>
        .centered-title {
            text-align: center;
        }
        
        .predict-button {
            display: inline-block;
            margin: 20px;
            background-color:#00ced1;
            color:Cyan ;
            border: none;
            padding: 14px 28px;
            text-align: center;
            font-size: 24px;
            cursor: pointer;
            border-radius: 8px;
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2);
    }
        </style>
        """,    
        unsafe_allow_html=True,
    )
    st.title("Dog Breed Classifier")

    # Upload an image for classification
    uploaded_image = st.sidebar.file_uploader("Upload a dog image", type=["jpg", "png", "jpeg"])

    if uploaded_image is not None:
        st.session_state.uploaded_image = uploaded_image
        image = tf.image.decode_image(uploaded_image.read(), channels=3)
        image = tf.image.resize(image, (224, 224))
        image = np.expand_dims(image, axis=0) / 255.0
        st.session_state.processed_image = image

    if st.session_state.uploaded_image is not None:
        st.image(st.session_state.uploaded_image, caption="Uploaded Image", use_column_width=True)

    if st.button("Classify", key="predict-button", help="Classify the uploaded image",use_container_width=True):
        with st.spinner("Classifying..."):
            # Add a progress bar with percentage
            progress_bar = st.progress(0)
            progress_text = st.empty()

            # Simulate some work with progress
            for i in range(100):
                progress_bar.progress(i + 1)
                progress_text.text(f"Processing: {i + 1}%")
                time.sleep(0.05)  # Simulate a delay

            progress_bar.empty()
            progress_text.empty()

            # Classification
            prediction = model.predict(st.session_state.processed_image)
            predicted_class = np.argmax(prediction, axis=-1)
            st.write(f"Predicted Breed: {class_names[str(predicted_class[0])]}")

            # Store probabilities for the Graphs section
            st.session_state.probabilities = prediction[0]

            # Update the navigation selection to "Graphs"
            st.session_state.selection = "Graphs"

# Graphs Section
if selection == "Graphs":
    if st.session_state.uploaded_image is not None:
        st.title("Prediction Probabilities")
        probabilities = st.session_state.probabilities
        fig, ax = plt.subplots()
        ax.barh(list(class_names.values()), probabilities, color='#235b66')
        ax.set_xlabel("Probability")
        ax.set_title("Prediction Probabilities for All Classes")
        st.pyplot(fig)
    else:
        st.write("Please upload an image and classify it to view graphs.")
