from flask import Flask, render_template, jsonify
import pandas as pd
import time 
from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier
import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime
from flask import Flask, render_template, Response,request
app = Flask(__name__)



ts = time.time()
date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")





@app.route('/')
def index():
    return render_template('index2.html')

@app.route('/update_count')
def update_count():
    count = int(time.time() % 100)  # Your count logic here
    if count == 0:
        result = "Count is zero"
    elif count % 3 == 0 and count % 5 == 0:
        result = "FizzBuzz"
    elif count % 3 == 0:
        result = "Fizz"
    elif count % 5 == 0:
        result = "Buzz"
    else:
        result = f"Count: {count}"
    return jsonify(result)

@app.route('/attendance')
def attendance():
    df = pd.read_csv(f"Attendance/Attendance_{date}.csv", index_col=0)  # Use the first column as index
    table = df.to_html(classes="table table-striped")
    return render_template('index2.html', table=table)
@app.route('/register_face', methods=['GET', 'POST'])
def register_face():
    if request.method == 'POST':
        name = request.form.get('name')
        video = cv2.VideoCapture(0)
        facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

        faces_data = []
        i = 0

        while True:
            ret, frame = video.read()

            # Check if the frame is successfully captured
            if not ret:
                print("Error: Couldn't read frame from the camera.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = facedetect.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                crop_img = frame[y:y + h, x:x + w, :]
                resized_img = cv2.resize(crop_img, (50, 50))
                if len(faces_data) <= 100 and i % 10 == 0:
                    faces_data.append(resized_img)
                i = i + 1
                cv2.putText(frame, str(len(faces_data)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 1)

            cv2.imshow("Frame", frame)
            k = cv2.waitKey(1)

            if k == ord('q') or len(faces_data) == 100:
                break

        video.release()
        cv2.destroyAllWindows()

        # Move the name saving part inside the loop
        if len(faces_data) > 0:
            faces_data = np.asarray(faces_data)
            faces_data = faces_data.reshape(100, -1)

            if 'names.pkl' not in os.listdir('data/'):
                names = [name] * 100
                with open('data/names.pkl', 'wb') as f:
                    pickle.dump(names, f)
            else:
                with open('data/names.pkl', 'rb') as f:
                    names = pickle.load(f)
                names = names + [name] * 100
                with open('data/names.pkl', 'wb') as f:
                    pickle.dump(names, f)
            if 'faces_data.pkl' not in os.listdir('data/'):
                with open('data/faces_data.pkl', 'wb') as f:
                    pickle.dump(faces_data, f)
            else:
                with open('data/faces_data.pkl', 'rb') as f:
                    faces = pickle.load(f)
                faces = np.append(faces, faces_data, axis=0)
                with open('data/faces_data.pkl', 'wb') as f:
                    pickle.dump(faces, f)
        
        result = f"Successfully registered {name}'s face!"
        return render_template("index2.html",   prediction=result) 
    register = "ok"
    return  render_template("index2.html",   register=register)

@app.route('/Take_attendance', methods=['GET', 'POST'])
def generate_frames():
    video=cv2.VideoCapture(0)
    facedetect=cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

    with open('data/names.pkl', 'rb') as w:
        LABELS=pickle.load(w)
    with open('data/faces_data.pkl', 'rb') as f:
        FACES=pickle.load(f)

    knn=KNeighborsClassifier(n_neighbors=5)
    knn.fit(FACES, LABELS)

    COL_NAMES = ['NAME', 'TIME', 'COURSE']

    while True:
        ret,frame=video.read()
        gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces=facedetect.detectMultiScale(gray, 1.3 ,5)
        for (x,y,w,h) in faces:
            crop_img=frame[y:y+h, x:x+w, :]
            resized_img=cv2.resize(crop_img, (50,50)).flatten().reshape(1,-1)
            output=knn.predict(resized_img)
            ts=time.time()
            date=datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
            timestamp=datetime.fromtimestamp(ts).strftime("%H:%M-%S")
            exist=os.path.isfile("Attendance/Attendance_" + date + ".csv")
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 1)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(50,50,255),2)
            cv2.rectangle(frame,(x,y-40),(x+w,y),(50,50,255),-1)
            cv2.putText(frame, str(output[0]), (x,y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 1)
            cv2.rectangle(frame, (x,y), (x+w, y+h), (50,50,255), 1)
            Course = "CSC419"
            attendance=[str(output[0]), str(timestamp), str(Course)]
        frame = frame
        cv2.imshow("Frame",frame)
        k=cv2.waitKey(1)
        if k==ord('o'):
            time.sleep(5)
            # Check if the file exists
            file_path = "Attendance/Attendance_" + date + ".csv"
            if os.path.exists(file_path):
                with open(file_path, "+a") as csvfile:
                    writer = csv.writer(csvfile)
                    with open(file_path, "r") as csvfile_read:
                        reader = csv.reader(csvfile_read)
                        num_rows = len(list(reader))  # Get the current number of rows
                        index = num_rows if num_rows > 0 else 0  # Calculate the starting index
                        writer.writerow([index + 1] + attendance)  # Write the row with the index
            else:
                with open(file_path, "+a") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['Index'] + COL_NAMES)  # Add an 'Index' column header
                    writer.writerow([1] + attendance)  # Write the first row with index 1

        if k==ord('o'):
            break
    video.release()
    cv2.destroyAllWindows()
    result = f"Attendance taken successfully"
    return render_template("index2.html",   prediction=result) 
if __name__ == '__main__':
    app.run()

