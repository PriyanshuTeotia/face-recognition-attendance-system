import cv2

# Load trained model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')

# Load Haar cascade
face_cascade = cv2.CascadeClassifier('cascade/haarcascade_frontalface_default.xml')

# Define font
font = cv2.FONT_HERSHEY_SIMPLEX

# Example names (edit later)
names = ['Unknown', 'Priyanshu Teotia', 'Harsh Teotia']  # ID=1 → Priyanshu Teotia

# Start camera
cam = cv2.VideoCapture(0)

print("Starting face recognition...")

while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

        # If confidence is less than 100 → match
        if confidence < 100:
            name = names[id]
            confidence_text = f"{round(100 - confidence)}%"
        else:
            name = "Unknown"
            confidence_text = f"{round(100 - confidence)}%"

        cv2.putText(img, str(name), (x+5,y-5), font, 1, (255,255,255), 2)
        cv2.putText(img, str(confidence_text), (x+5,y+h-5), font, 1, (255,255,0), 1)

    cv2.imshow('Face Recognition', img)

    if cv2.waitKey(1) == 27:
        break

cam.release()
cv2.destroyAllWindows()