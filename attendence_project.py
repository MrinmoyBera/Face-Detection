import face_recognition
import cv2
import numpy as np
import csv 
from datetime import datetime
font = cv2.FONT_HERSHEY_SIMPLEX

# Initialize video capture
video_capture = cv2.VideoCapture(0)


# Load and encode known images
known_face_encodings = []
for i in range(1,24) :
    if i < 10 :
        img = face_recognition.load_image_file(f"C:\\Users\\Mrinmoy Bera\\Desktop\\project\\face_detection\\crs230{i}.jpeg")
        encoding = face_recognition.face_encodings(img)[0]
        known_face_encodings.append(encoding)
    else :
       img = face_recognition.load_image_file(f"C:\\Users\\Mrinmoy Bera\\Desktop\\project\\face_detection\\crs23{i}.jpeg") 
       encoding = face_recognition.face_encodings(img)[0]
       known_face_encodings.append(encoding)
 
#known names    
known_face_names = []  
for i in range(1,24) :
    if i<10 :
        known_face_names.append(f"crs230{i}")
    else:
        known_face_names.append(f"crs23{i}")
    
  
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
    #capture each frame from webcam
    ret, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]
    
    #convert the frame from BGR to RGB
    code = cv2.COLOR_BGR2RGB
    rgb_small_frame = cv2.cvtColor(rgb_small_frame, code)

    if s:
        #face locations of the resize frame
        face_locations = face_recognition.face_locations(rgb_small_frame)
        print("Face locations:", face_locations)  # Debugging line

        if face_locations:
            try:
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                face_names = []

                for face_encoding in face_encodings:
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                    name = ""
                    #distances between frame and the images that are store in data base
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    #Store the index of the nearest image in the data base
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        #store the name of matched face
                        name = known_face_names[best_match_index]
                        
                        # for draw a rectangle on the selected part from the frame
                        frame_locations = face_recognition.face_locations(frame)
                        frame_size = list(frame_locations[0])
                        x1, y1, x2, y2 = frame_size[:4]
                        img = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)                    
                        img = cv2.putText(frame, name, (x1, y1 - 10), font, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

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
