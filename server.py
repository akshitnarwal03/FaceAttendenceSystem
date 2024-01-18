from flask import Flask, render_template, request, redirect, url_for, Markup
import cv2
import os
import face_recognition
import pickle
import dlib
import numpy as np
import base64
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
import time
import datetime
from datetime import datetime
 
# Create datetime string
datetime_obj = datetime.now()
 
# extract the time from datetime_obj
date = datetime_obj.date()

app = Flask(__name__)

face_detector = dlib.get_frontal_face_detector()
apps_script_url = 'https://script.google.com/a/gmail.com/macros/s/AKfycbwhEF3_u5-g_x7Yte_bR2PiJR0vLmhA6tpz1Gv4AoOXS1HARXhUjoNJjt2BvDL1FJPt/exec'

scope = ['https://spreadsheets.google.com/feeds',
         'https://www.googleapis.com/auth/drive']

creds = ServiceAccountCredentials.from_json_keyfile_name('astute-dreamer-359413-2680db7e5c84.json', scope)
client = gspread.authorize(creds)

# Get the instance of the Spreadsheet
sheet = client.open('attendence')

# Get the first sheet of the Spreadsheet
sheet_instance = sheet.get_worksheet(0)
col = 1

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])


def upload_image():
    detected_names = []
    global col
    c = 0
    if 'image' in request.files:
        image = request.files['image']
        academic_group = request.form['academic_group']  # Get the selected academic group

        if image.filename != '':
            filename = image.filename
            print(filename)
            image_bytes = image.read()
            image_np = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

            encodings_file_path = f'encodings/{academic_group}.pkl'
            images_folder_path = f'classes/{academic_group}'

            with open(encodings_file_path, 'rb') as f:
                encodeListKnown = pickle.load(f)

            classNames = [os.path.splitext(cl)[0] for cl in os.listdir(images_folder_path)]

            face_locations = face_detector(img, 1)
            confidence_threshold = 0.55

            for face in face_locations:
                l, t, r, b = face.left(), face.top(), face.right(), face.bottom()
                face = img[t:b, l:r]

                face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

                face_encodings = face_recognition.face_encodings(face_rgb)

                if len(face_encodings) > 0:
                    encodeFace = face_encodings[0]

                    matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
                    faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
                    matchIndex = np.argmin(faceDis)

                    if faceDis[matchIndex] < confidence_threshold:
                        if matches[matchIndex]:
                            name = classNames[matchIndex].upper()
                            detected_names.append(name)
                            cv2.circle(img, (int((l + r) // 2), int((t + b) // 2)), 0, (0, 255, 0), 20)
                            if not os.path.exists(f"./save_group/group/{filename.split('.')[0]}/detected"):
                                os.makedirs(f"./save_group/group/{filename.split('.')[0]}/detected")
                            cv2.imwrite(f"./save_group/group/{filename.split('.')[0]}/detected/{name}.jpg", face_rgb)
                    else:
                        cv2.circle(img, (int((l + r) // 2), int((t + b) // 2)), 0, (0, 0, 255), 20)
                        if not os.path.exists(f"./save_group/group/{filename.split('.')[0]}/not_detected"):
                            os.makedirs(f"./save_group/group/{filename.split('.')[0]}/not_detected")
                        cv2.imwrite(f"./save_group/group/{filename.split('.')[0]}/not_detected/unknown{c}.jpg", face_rgb)
                        c = c + 1

            cv2.imwrite(f"./save_group/{filename}", img)
            print(detected_names)
            df = pd.DataFrame({'roll': detected_names})
            sheet_instance.update_cell(1, col, str(date))
            for i, name in enumerate(detected_names):
                sheet_instance.update_cell(i + 2, col, name)        

            # Encode the processed image in base64 format for displaying in HTML
            img_bytes = cv2.imencode(".jpg", img)[1].tobytes()
            img_base64 = base64.b64encode(img_bytes).decode('utf-8')
            image_data_uri = f"data:image/jpeg;base64,{img_base64}"
            col+=1
            # Return a response with the result
            return render_template('result.html', image_path=image_data_uri, detected_names=detected_names, recognised=len(detected_names), not_recognised=c)


    # Handle the case when no image is uploaded or there is an error
    return redirect(url_for('home'))


if __name__ == '__main__':
    app.run(debug=True)
