from flask import Flask, request, render_template, Response, url_for, redirect,send_file,jsonify,send_from_directory,flash
import pywhatkit as kit
import pyautogui as py
import time
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import datetime
from serpapi import GoogleSearch
import pyttsx3

from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from comtypes import CLSCTX_ALL, CoInitialize, CoUninitialize
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

import requests
import base64


import threading
import cv2
import numpy as np


from werkzeug.utils import secure_filename
import os
from io import BytesIO


from PIL import Image
import io

import joblib

import pandas as pd
import numpy as np
import warnings

#Flask App Initialization 
app = Flask(__name__)

















##################################################      Function to send Whatsapp message     ##############################################
def send_whatsapp_message(phone_number, message, hour, minute):
    try:
        kit.sendwhatmsg(phone_number, message, hour, minute)
        return f"Message scheduled to be sent to {phone_number} at {hour:02d}:{minute:02d}"
    except Exception as e:
        return str(e)





################################################      Function To SEnd SMS By connecting Phone     #######################################
def send_message_via_phone_link(phone, msg): 
    try:
        # Open the Phone Link app
        py.press('win')
        time.sleep(3)
        py.typewrite("Phone Link")
        time.sleep(5)
        py.press('enter')
        time.sleep(8)  # Adjust based on system performance

        # Locate and click the compose button
        compose = py.locateOnScreen("static/compose.png", confidence=0.8)
        if compose:
            py.click(compose)
        else:
            return "Compose button not found."
        time.sleep(2)

        # Locate and click the number field
        number_field = py.locateOnScreen("static/number.png", confidence=0.8)
        if number_field:
            py.click(number_field)
            py.typewrite(phone)
            time.sleep(3)
            py.press('enter')
            time.sleep(6)
        else:
            return "Number field not found."

        # Locate and click the message field
        message_field = py.locateOnScreen("static/message.png", confidence=0.8)
        py.click(message_field)
        py.typewrite(msg)
        time.sleep(6)
        # Locate and click the send button
        send_button = py.locateOnScreen('static/send.png', confidence=0.8)
        if send_button:
            py.click(send_button)
        else:
            return "Send button not found."

        return "Your message has been sent successfully."
    except Exception as e:
        return str(e)









#################################################   Function for Sending Email      ####################################################
def send_email(receiver_email, subject, message):
    try:
        email = "iitzvikas@gmail.com"
        text = f"Subject : {subject}\n\n{message}"
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(email, "rhbecciuspztdpyq")
        server.sendmail(email, receiver_email, text)
        server.quit()
        date = datetime.date.today().strftime("%Y-%m-%d")
        return f"Email sent to {receiver_email} successfully on {date}"
    except Exception as e:
        return str(e)
    









###############################################     Function for Sending Email in bulk     #############################################
def send_bulk_emails(email_list, subject, message):
    # Email configuration
    sender_email = "iitzvikas@gmail.com"  # Replace with your email
    sender_password = "rhbecciuspztdpyq"      # Replace with your password or app password

    # Connect to SMTP server
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
    except Exception as e:
        print(f"Failed to connect to SMTP server: {str(e)}")
        return f"Failed to connect to SMTP server: {str(e)}"

    for receiver_email in email_list:
        try:
            # Create a new message container for each recipient
            msg = MIMEMultipart()
            msg['From'] = sender_email
            msg['To'] = receiver_email
            msg['Subject'] = subject

            # Add message body
            msg.attach(MIMEText(message, 'plain'))

            # Send the email
            server.sendmail(sender_email, receiver_email, msg.as_string())
            print(f"Email sent to {receiver_email} successfully")
        except Exception as e:
            print(f"Error sending email to {receiver_email}: {str(e)}")
    
    server.quit()










############################################     Function for Query Search on Google     ################################################
def google_search(query):
    api_key = "f9dc3a810e1e3f15cec84beff8232b327d66ed776ed0c3f0c1f980992994b62d"
    params ={
        "engine": "google",
        "q": query,
        "api_key": api_key
        }
    search = GoogleSearch(params)
    results = search.get_dict()
    top_results = results.get('organic_results', [])[:5]
    return top_results





#############################################    Function for Speaking String   ########################################################

# Function to speak using pyttsx3
def speak(text, rate=150):
    engine = pyttsx3.init()
    engine.setProperty('rate', rate)
    engine.say(text)
    engine.runAndWait()

# Function to print and speak
def print_and_speak(text):
    print(text)
    speak(text)






###############################################   Function For Controlling or Setting Volume   ############################################
# Function to set volume
def set_volume(volume_level):
    try:
        CoInitialize()
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(
            IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        volume = cast(interface, POINTER(IAudioEndpointVolume))
        volume_level = float(volume_level)
        # Volume level must be between 0.0 and 1.0
        volume.SetMasterVolumeLevelScalar(volume_level / 100, None)
        CoUninitialize()
        return f"Volume set to {volume_level}%"
    except Exception as e:
        CoUninitialize()
        return str(e)









#################################################     Function for finding Geo-Cordinates     #############################################
# Function to get geo coordinates and location
def get_geo_location(query=''):
    try:
        if query:
            response = requests.get(f"https://ipinfo.io/{query}")
        else:
            response = requests.get("https://ipinfo.io")
        data = response.json()
        location = data.get('loc', 'Location not found')
        city = data.get('city', 'City not found')
        region = data.get('region', 'Region not found')
        country = data.get('country', 'Country not found')
        return f"Coordinates: {location}, Location: {city}, {region}, {country}"
    except Exception as e:
        return str(e)







##############################################     Function for Cropping Face during live video    ####################################

camera = None  # Global variable to hold the camera instance

def detect_and_generate():
    global camera
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    camera = cv2.VideoCapture(0)

    while True:
        ret, frame = camera.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        cropped_faces = []
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            face = frame[y:y+h, x:x+w]
            cropped_faces.append(face)

        if cropped_faces:
            # Resize all faces to a fixed height (adjust as needed)
            max_height = max(face.shape[0] for face in cropped_faces)
            for i in range(len(cropped_faces)):
                face = cropped_faces[i]
                new_height = max_height
                new_width = int(face.shape[1] * (new_height / face.shape[0]))
                cropped_faces[i] = cv2.resize(face, (new_width, new_height))

            combined_faces = cv2.hconcat(cropped_faces)
            combined_faces = cv2.resize(combined_faces, (frame.shape[1], combined_faces.shape[0]))
            frame_with_faces = cv2.vconcat([frame, combined_faces])
        else:
            frame_with_faces = frame

        ret, jpeg = cv2.imencode('.jpg', frame_with_faces)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    camera.release()








########################################    Function For Applying Filters     #########################################################

# Initialize Cascade Classifiers
eye_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')

# Load Filters
filters = {
    'none': None,  # No filter
    'mask': cv2.imread('static/filters/mask.png', cv2.IMREAD_UNCHANGED),
    """'glass': cv2.imread('static/filters/glass.png', cv2.IMREAD_UNCHANGED),"""
    'mask1': cv2.imread('static/filters/mask1.png', cv2.IMREAD_UNCHANGED),
    'melon': cv2.imread('static/filters/melon.png', cv2.IMREAD_UNCHANGED),
    'star': cv2.imread('static/filters/star.png', cv2.IMREAD_UNCHANGED)
}

# Global Variables
cap = None
selected_filter = None

def generate_filtered_video():
    global cap, selected_filter
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.1, 4)
        
        for (x, y, w, h) in faces:
            if selected_filter is not None:
                selected_filter_resized = cv2.resize(selected_filter, (w, h))
                
                for i in range(y, y + h):
                    for j in range(x, x + w):
                        if selected_filter_resized[i - y, j - x, 3] != 0:  # Check alpha channel for transparency
                            frame[i, j] = selected_filter_resized[i - y, j - x, :3]  # Assign RGB values

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


        



###############################       Function to apply  a Filter on image     ###################################################

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def apply_sepia_filter(img):
    sepia_filter = np.array([[0.272, 0.534, 0.131],
                             [0.349, 0.686, 0.168],
                             [0.393, 0.769, 0.189]])
    sepia_img = cv2.transform(img, sepia_filter)
    sepia_img = np.clip(sepia_img, 0, 255)
    return sepia_img

def apply_color_filter(img, color):
    b, g, r = color
    colored_img = np.zeros_like(img)
    colored_img[:, :, 0] = b
    colored_img[:, :, 1] = g
    colored_img[:, :, 2] = r
    filtered_img = cv2.addWeighted(img, 0.5, colored_img, 0.5, 0)
    return filtered_img














def detect_faces(image):
    # Load the pre-trained Haar Cascade model for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Count the number of faces detected
    num_faces = len(faces)
    
    return image, num_faces
































# Function to capture an image from the webcam
def capture_image():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return None

    ret, frame = cap.read()
    if not ret:
        print("Error: Could not capture image.")
        cap.release()
        return None

    cap.release()
    return frame











from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import os


# Automatic Data Processing Function
def automatic_data_processing(df):
    # Handling missing values
    num_imputer = SimpleImputer(strategy='mean')
    df[df.select_dtypes(include=[np.number]).columns] = num_imputer.fit_transform(df.select_dtypes(include=[np.number]))

    cat_imputer = SimpleImputer(strategy='most_frequent')
    df[df.select_dtypes(include=[object]).columns] = cat_imputer.fit_transform(df.select_dtypes(include=[object]))

    # One-Hot Encoding
    df = pd.get_dummies(df, drop_first=True)

    # Feature Scaling
    scaler = StandardScaler()
    df[df.columns] = scaler.fit_transform(df)

    return df








###############     Routing Code Start Here ----------------------------------------------->>



#Routing on Index page by Default
@app.route('/')
def index():
    return render_template('index.html')





#Routing On Whatsapp template 
@app.route('/whatsapp', methods=['GET', 'POST'])
def whatsapp():
    if request.method == 'POST':
        phone_number = request.form['phone_number']
        message = request.form['message']
        hour = int(request.form['hour'])
        minute = int(request.form['minute'])

        result = send_whatsapp_message(phone_number, message, hour, minute)
        return render_template('result.html', result=result)

    return render_template('whatsapp.html')





#Routing on Phone_link template
@app.route('/phone_link', methods=['GET', 'POST'])
def phone_link():
    if request.method == 'POST':
        phone = request.form['phone']
        msg = request.form['msg']

        result = send_message_via_phone_link(phone, msg)
        return render_template('result.html', result=result)

    return render_template('phone_link.html')






#Routing on Email template 
@app.route('/email', methods=['GET', 'POST'])
def email():
    if request.method == 'POST':
        receiver_email = request.form['receiver_email']
        subject = request.form['subject']
        message = request.form['message']

        result = send_email(receiver_email, subject, message)
        return render_template('result.html', result=result)
    return render_template('email.html')






#Routing On Send_Bulk_Email template
@app.route('/send_bulk_email', methods=['GET', 'POST'])
def send_bulk_email():
    if request.method == 'POST':
        # Retrieve form data
        email_list = request.form.get('email_list').split(',')
        email_list = [email.strip() for email in email_list]  # Strip whitespace from each email
        subject = request.form['subject']
        message = request.form['message']

        # Send bulk emails
        result = send_bulk_emails(email_list, subject, message)

        # Prepare result message
        if result:
            result_message = result
        else:
            result_message = f"Bulk emails sent to {', '.join(email_list)} successfully."

        return render_template('result.html', result=result_message)

    return render_template('send_bulk_email.html')








#Routing on Google_search template
@app.route('/google_search', methods=['GET', 'POST'])
def search():
    if request.method == 'POST':
        query = request.form['query']
        results = google_search(query)
        return render_template('search_results.html', results=results)
    return render_template('google_search.html')





#Routing on Speak template
@app.route('/speak', methods=['GET', 'POST'])
def speak_text():
    if request.method == 'POST':
        text = request.form['text']
        print_and_speak(text)
        return render_template('result.html', result=f'Text "{text}" spoken successfully')
    return render_template('speak.html')







#Routing on Volume templatee
@app.route('/volume', methods=['GET', 'POST'])
def volume():
    if request.method == 'POST':
        volume_level = request.form['volume_level']
        result = set_volume(volume_level)
        return render_template('result.html', result=result)
    return render_template('volume.html')







# Routing On Geo-location template
@app.route('/geo_location', methods=['GET', 'POST'])
def geo_location():
    if request.method == 'POST':
        result = get_geo_location()
        return render_template('result.html', result=result)
    return render_template('geo_location.html')

@app.route('/geo_location', methods=['GET', 'POST'])
def geo_location_route():
    if request.method == 'POST':
        location_input = request.form.get('location_input', '')
        result = get_geo_location(location_input)
        return render_template('result.html', result=result)
    return render_template('geo_location.html')








# Routing on Face-detection1(crop) template
@app.route('/face_detection')
def face_detection():
    return render_template('face_detection.html')

@app.route('/video_feed')
def video_feed():
    return Response(detect_and_generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_camera')
def stop_camera():
    global camera
    if camera is not None:
        camera.release()
    return "Camera stopped."









# Routing on Face filter template
@app.route('/face_detection_filters')
def face_detection_filters():
    return render_template('face_detection_filters.html')

@app.route('/video_feed_with_filters')
def video_feed_with_filters():
    global cap
    if cap is None:
        cap = cv2.VideoCapture(0)
    
    return Response(generate_filtered_video(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/select_video_filter/<filter_name>')
def select_video_filter(filter_name):
    global selected_filter
    if filter_name in filters:
        selected_filter = filters[filter_name]
        print(f"Filter selected: {filter_name}")
    return redirect(url_for('face_detection_filters'))

@app.route('/stop_filtered_camera')
def stop_filtered_camera():
    global cap
    if cap is not None:
        cap.release()
        cap = None
    return redirect(url_for('index'))
    







#Routing for Filter image or apply

@app.route('/upload_image', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            img = Image.open(file.stream)
            img.save(os.path.join(app.config['UPLOAD_FOLDER'], 'original_image.jpg'))
            return redirect(url_for('filter_image'))
    return render_template('upload_image.html')

@app.route('/filter_image')
def filter_image():
    filters = ['none', 'sepia', 'red', 'green', 'blue']
    return render_template('filter_image.html', filters=filters)

@app.route('/apply_filter', methods=['POST'])
def apply_filter():
    filter_name = request.json.get('filter')
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], 'original_image.jpg')
    img = cv2.imread(img_path)

    if filter_name == 'sepia':
        img = apply_sepia_filter(img)
    elif filter_name == 'red':
        img = apply_color_filter(img, (0, 0, 255))
    elif filter_name == 'green':
        img = apply_color_filter(img, (0, 255, 0))
    elif filter_name == 'blue':
        img = apply_color_filter(img, (255, 0, 0))

    processed_img_path = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_image.jpg')
    cv2.imwrite(processed_img_path, img)
    _, buffer = cv2.imencode('.jpg', img)
    img_str = base64.b64encode(buffer).decode('utf-8')

    return jsonify({'image': img_str})

@app.route('/uploads/<filename>')
def send_uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)









####################################################     Linear Regression       #########################################################

import joblib
import warnings
# Load your model
brain = joblib.load('saved_model.pkl')


@app.route('/marks_prediction', methods=['GET', 'POST'])
def marks_prediction():
    prediction = None
    if request.method == 'POST':
        hours = float(request.form['hours'])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            prediction = brain.predict([[hours]])[0].round(2)
    return render_template('marks_prediction.html', prediction=prediction)
















#face detection
@app.route('/detect_faces', methods=['GET', 'POST'])
def detect_faces_route():
    if request.method == 'POST':
        # Check if the POST request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        file = request.files['file']
        
        # If the user does not select a file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        if file:
            # Read the image file
            npimg = np.fromfile(file, np.uint8)
            image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

            # Detect faces in the image
            result_image, num_faces = detect_faces(image)

            # Save the result image with rectangles around faces
            cv2.imwrite('static/result_faces.jpg', result_image)

            return render_template('detect_faces_result.html', num_faces=num_faces)
    
    return render_template('detect_faces.html')














####################################################  Multi-Linear Regression    #########################################################


#IPL score prediction
model = joblib.load('ipl_score_prediction_model.pkl')

@app.route('/predict_form')
def predict_form():
    return render_template('ipl_prediction.html')

# Route for predicting IPL score and displaying result
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect user input from the form
        runs_scored = float(request.form['runs_scored'])
        wickets_lost = float(request.form['wickets_lost'])
        overs_played = float(request.form['overs_played'])
        runs_conceded = float(request.form['runs_conceded'])
        extras = float(request.form['extras'])
        boundaries_hit = float(request.form['boundaries_hit'])
        sixes_hit = float(request.form['sixes_hit'])
        partnerships = float(request.form['partnerships'])

        # Make prediction
        input_data = np.array([[runs_scored, wickets_lost, overs_played, runs_conceded, extras, boundaries_hit, sixes_hit, partnerships]])
        prediction = model.predict(input_data)[0]

        # Render the prediction result on the same page
        return render_template('ipl_prediction.html', prediction=prediction, 
                               runs_scored=runs_scored, wickets_lost=wickets_lost,
                               overs_played=overs_played, runs_conceded=runs_conceded,
                               extras=extras, boundaries_hit=boundaries_hit,
                               sixes_hit=sixes_hit, partnerships=partnerships)
    except Exception as e:
        return jsonify({'error': str(e)})
    





###########################################    Image Processing     ##################################################

import google.generativeai as genai
from IPython.display import Markdown
from PIL import Image


GOOGLE_API_KEY = 'AIzaSyCBni1ZR0Bx3JzkkZNcQaVcO4b5YxIwFkA'
genai.configure(api_key=GOOGLE_API_KEY)

# Function to handle Google Generative AI prompts
def handle_prompt(image_path, prompt_text):
    try:
        # Upload the image file to Google Generative AI
        sample_file = genai.upload_file(path=image_path, display_name=os.path.basename(image_path))

        # Choose a Gemini API model
        model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest")

        # Prompt the model with the text and the uploaded image
        response = model.generate_content([sample_file, prompt_text])

        return response.text

    except Exception as e:
        print(f"Error handling prompt: {e}")
        return str(e)

# Route for Google Generative AI prompt form and processing
@app.route('/generate_prompt', methods=['GET', 'POST'])
def generate_prompt():
    if request.method == 'POST':
        file = request.files['file']
        prompt_text = request.form['prompt_text']

        if file and prompt_text:
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)

            response_text = handle_prompt(filename, prompt_text)

            return render_template('prompt_result.html', prompt_text=prompt_text, response_text=response_text)

    return render_template('generate_prompt.html')

# Route for displaying the result
@app.route('/prompt_result')
def prompt_result():
    return render_template('prompt_result.html')





#############################################     Video  Process   #################################################################
  
# Configure Google Generative AI with your API key
GOOGLE_API_KEY = 'AIzaSyCBni1ZR0Bx3JzkkZNcQaVcO4b5YxIwFkA'
genai.configure(api_key=GOOGLE_API_KEY)

# Function to handle video processing with Google Generative AI
def handle_video_processing(video_path, prompt_text):
    try:
        # Upload the video file to Google Generative AI
        sample_file = genai.upload_file(path=video_path, display_name=os.path.basename(video_path))

        # Choose a Gemini API model
        model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest")

        # Prompt the model with the text and the uploaded video
        response = model.generate_content([sample_file, prompt_text])

        return response.text

    except Exception as e:
        print(f"Error handling video processing: {e}")
        return str(e)

# Route for video processing form and processing
@app.route('/process_video', methods=['GET', 'POST'])
def process_video():
    if request.method == 'POST':
        video_file = request.files['video_file']
        prompt_text = request.form['prompt_text']

        if video_file and prompt_text:
            video_filename = os.path.join(app.config['UPLOAD_FOLDER'], video_file.filename)
            video_file.save(video_filename)

            response_text = handle_video_processing(video_filename, prompt_text)

            return render_template('video_processing_result.html', prompt_text=prompt_text, response_text=response_text)

    return render_template('process_video.html')

# Route for displaying the result
@app.route('/video_processing_result')
def video_processing_result():
    return render_template('video_processing_result.html')




#prediction og Iris dataset

from sklearn.datasets import load_iris
iris = load_iris()
# Prediction function
def predict_iris(sepal_length, sepal_width, petal_length, petal_width):
    model = joblib.load('model.joblib')
    prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    return iris.target_names[prediction][0]


@app.route('/predict_iris', methods=['GET', 'POST'])
def predict_iris_route():
    prediction = None
    if request.method == 'POST':
        try:
            sepal_length = float(request.form['sepal_length'])
            sepal_width = float(request.form['sepal_width'])
            petal_length = float(request.form['petal_length'])
            petal_width = float(request.form['petal_width'])
            prediction = predict_iris(sepal_length, sepal_width, petal_length, petal_width)
        except ValueError:
            prediction = "Invalid input. Please enter numeric values."
    return render_template('iris_predict.html', prediction=prediction)















###   Prediction of cat or dog
from tensorflow.keras.preprocessing import image

import tensorflow as tf
# Load the saved model
cats_dogs_model = tf.keras.models.load_model('cats_dogs_cnn_model.h5')

@app.route('/predict_cat_dog', methods=['GET', 'POST'])
def predict_cat_dog():
    result = None
    if request.method == 'POST':
        file = request.files['file']
        if file:
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)

            img = image.load_img(file_path, target_size=(128, 128))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            prediction = cats_dogs_model.predict(img_array)
            result = 'Dog' if prediction[0][0] > 0.5 else 'Cat'

    return render_template('predict_cat_dog.html', result=result)







# Prediction based on titanic dataset

from sklearn.preprocessing import LabelEncoder


# Load the saved model, label encoder, and scaler
model = tf.keras.models.load_model('titanic_cnn_model.h5')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

@app.route('/predict_titanic', methods=['GET', 'POST'])
def predict_titanic():
    result = None
    if request.method == 'POST':
        Pclass = int(request.form['Pclass'])
        Sex = request.form['Sex']
        Age = float(request.form['Age'])
        SibSp = int(request.form['SibSp'])

        # Encode the Sex feature
        Sex = label_encoder.transform([Sex])[0]

        # Create the feature array
        features = np.array([[Pclass, Sex, Age, SibSp]])

        # Scale the features
        features_scaled = scaler.transform(features)

        # Predict the survival
        prediction = model.predict(features_scaled)
        prediction = (prediction > 0.5).astype(int)

        result = 'Survived' if prediction[0][0] == 1 else 'Did not survive'

    return render_template('predict_titanic.html', result=result)










#  Text sentiment analysis


from textblob import TextBlob


# Route for sentiment analysis
@app.route('/sentiment_analysis', methods=['GET', 'POST'])
def sentiment_analysis():
    sentiment = None
    if request.method == 'POST':
        text = request.form['text']
        blob = TextBlob(text)
        sentiment = blob.sentiment.polarity
        if sentiment > 0:
            sentiment = 'Positive'
        elif sentiment < 0:
            sentiment = 'Negative'
        else:
            sentiment = 'Neutral'
    return render_template('sentiment_analysis.html', sentiment=sentiment)












# Combined Image 


@app.route('/capture_images', methods=['GET', 'POST'])
def capture_images():
    if request.method == 'POST':
        # Capture the first image
        image1 = capture_image()
        if image1 is None:
            return "Failed to capture the first image."

        # Save the first image temporarily
        first_image_path = os.path.join('static', 'first_image.jpg')
        cv2.imwrite(first_image_path, image1)

        return render_template('capture_second.html', first_image_path=first_image_path)
    return render_template('capture.html')

@app.route('/capture_second', methods=['POST'])
def capture_second():
    first_image_path = request.form.get('first_image_path')

    # Capture the second image
    image2 = capture_image()
    if image2 is None:
        return "Failed to capture the second image."

    # Load the first image
    image1 = cv2.imread(first_image_path)

    # Resize the second image to be smaller than the first image
    height1, width1 = image1.shape[:2]
    scale_factor = 0.25  # Adjust this factor as needed
    image2_resized = cv2.resize(image2, (int(width1 * scale_factor), int(height1 * scale_factor)))

    # Overlay the second image onto the first image at the top-left corner
    y_offset, x_offset = 0, 0
    y1, y2 = y_offset, y_offset + image2_resized.shape[0]
    x1, x2 = x_offset, x_offset + image2_resized.shape[1]

    alpha_s = image2_resized[:, :, 2] / 255.0
    alpha_l = 1.0 - alpha_s

    for c in range(0, 3):
        image1[y1:y2, x1:x2, c] = (alpha_s * image2_resized[:, :, c] + alpha_l * image1[y1:y2, x1:x2, c])

    # Save the combined image
    combined_image_path = os.path.join('static', 'combined_image.jpg')
    cv2.imwrite(combined_image_path, image1)

    return render_template('capture_result.html', image_path='combined_image.jpg')









# Route for Automatic Data Processing
@app.route('/automatic_data_processing', methods=['GET', 'POST'])
def automatic_data_processing_route():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            df = pd.read_csv(file)
            processed_df = automatic_data_processing(df)

            # Ensure the 'uploads' directory exists
            uploads_dir = os.path.join(os.getcwd(), 'uploads')
            if not os.path.exists(uploads_dir):
                os.makedirs(uploads_dir)

            # Save the processed file
            processed_file_path = os.path.join(uploads_dir, 'processed_data.csv')
            processed_df.to_csv(processed_file_path, index=False)

            return render_template('automatic_data_processing.html',
                                   preview=processed_df.head().to_html()
                                )
    return render_template('automatic_data_processing.html')




if __name__ == '__main__':
    app.run(debug=True)