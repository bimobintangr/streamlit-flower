import tensorflow as tf
import streamlit as st

# Function to load the model
def load_model():
    model = tf.keras.models.load_model('my_model2.hdf5')
    return model

# Function to manage login state
def login():
    if st.session_state.get("logged_in"):
        return True
    else:
        return False

# Function to authenticate user
def authenticate(username, password):
    if username == "admin" and password == "password":
        st.session_state.logged_in = True
        st.success("Login successful")
    else:
        st.error("Invalid username or password")

# Display the login page
def login_page():
    
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        authenticate(username, password)

# Display the logout button
def logout():
    st.session_state.logged_in = False
    st.success("Logged out successfully")

# Function to calculate total price
def hitung_total_harga(bunga, jumlah):
    harga_bunga = {
        'daisy': 5,       # Harga daisy per bunga
        'dandelion': 7,   # Harga dandelion per bunga
        'roses': 10,      # Harga roses per bunga
        'sunflowers': 12, # Harga sunflowers per bunga
        'tulips': 8       # Harga tulips per bunga
    }
    harga_per_bunga = harga_bunga.get(bunga, 0)
    total_harga = harga_per_bunga * jumlah
    return total_harga

# Main application
def main():
    st.title("Flower Classification")

    file = st.file_uploader("Please upload an image file", type=["jpg", "png"])

    import cv2
    from PIL import Image, ImageOps
    import numpy as np

    st.set_option('deprecation.showfileUploaderEncoding', False)

    class_names = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

    def import_and_predict(image_data, model):
        size = (180, 180)
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = np.asarray(image)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_reshape = img[np.newaxis, ...]
        prediction = model.predict(img_reshape)
        return prediction

    if file is None:
        st.text("Please upload an image file")
    else:
        image = Image.open(file)
        st.image(image, use_column_width=True)
        predictions = import_and_predict(image, model)
        score = tf.nn.softmax(predictions[0])
        predicted_class = class_names[np.argmax(score)]
        st.write("This image most likely belongs to {} with a {:.2f} percent confidence.".format(predicted_class, 100 * np.max(score)))

        # Form untuk input jumlah bunga yang ingin dibeli
        jumlah_bunga = st.number_input('Masukkan jumlah bunga yang ingin dibeli', min_value=1, step=1)

        # Tombol untuk submit jumlah bunga
        if st.button('Hitung Total Harga'):
            total_harga = hitung_total_harga(predicted_class, jumlah_bunga)
            st.write("Total Harga:", total_harga, "IDR")

# Initialize session state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# Show login page or main application based on login state
if login():
    with st.spinner('Model is being loaded..'):
        model = load_model()
    main()
    if st.button("Logout"):
        logout()
else:
    login_page()
