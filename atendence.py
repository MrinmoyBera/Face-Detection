import face_recognition
import cv2
import numpy as np
import csv 
from datetime import datetime

# Initialize video capture
video_capture = cv2.VideoCapture(0)

# Load and encode known images
me_img = face_recognition.load_image_file('D:\\movie\\me.jpg')
me_encoding = face_recognition.face_encodings(me_img)[0]

apj_img = face_recognition.load_image_file("C:\\Users\\Mrinmoy Bera\\Downloads\\APJ.jpg")
apj_encoding = face_recognition.face_encodings(apj_img)[0]

tata_img = face_recognition.load_image_file("C:\\Users\\Mrinmoy Bera\\Downloads\\Ratan-Tata.jpg")
tata_encoding = face_recognition.face_encodings(tata_img)[0]

jobs_img = face_recognition.load_image_file("C:\\Users\\Mrinmoy Bera\\Downloads\\steve jobs.jfif")
jobs_encoding = face_recognition.face_encodings(jobs_img)[0]

musk_img = face_recognition.load_image_file("C:\\Users\\Mrinmoy Bera\\Downloads\\Elon_Musk.jpg")
musk_encoding = face_recognition.face_encodings(musk_img)[0]

# Store known face encodings and names
known_face_encodings = [me_encoding, apj_encoding, tata_encoding, jobs_encoding, musk_encoding]
known_face_names = ["Mrinmoy", "APJ", "Tata", "Jobs", "Musk"]
students = known_face_names.copy()

# Initialize variables
face_locations = []
face_encodings = []
face_names = []
s = True
now = datetime.now()
current_date = now.strftime("%Y-%m-%d")
f = open(current_date+'.csv', 'w+', newline='')
lnwriter = csv.writer(f)

while True:
    ret, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]
    code = cv2.COLOR_BGR2RGB
    rgb_small_frame = cv2.cvtColor(rgb_small_frame, code)

    if s:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        print("Face locations:", face_locations)  # Debugging line

        if face_locations:
            try:
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                face_names = []

                for face_encoding in face_encodings:
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                    name = ""
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]

                    face_names.append(name)
                    if name in known_face_names:
                        if name in students:
                            students.remove(name)
                            print(students)
                            current_time = now.strftime("%H-%M-%S")
                            lnwriter.writerow([name, current_time])
            except Exception as e:
                print(f"An error occurred: {e}")

    cv2.imshow("Attendance System", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
f.close()
