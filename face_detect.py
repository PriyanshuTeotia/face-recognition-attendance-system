import cv2

print("Starting program...")

face_cascade = cv2.CascadeClassifier('cascade/haarcascade_frontalface_default.xml')

if face_cascade.empty():
    print("Error: Cascade file not loaded")
    exit()

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Camera not opening")
    exit()

print("Camera started successfully")

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame")
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    cv2.imshow('Face Detection', frame)
    
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()