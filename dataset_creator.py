import cv2

# Initialize camera
cam = cv2.VideoCapture(0)

# Load Haar Cascade
face_detector = cv2.CascadeClassifier('cascade/haarcascade_frontalface_default.xml')

# Ask user ID
face_id = input("Enter User ID: ")

print("Look at camera... Capturing faces")

count = 0

while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        count += 1
        
        # Save face image
        cv2.imwrite(f"dataset/User.{face_id}.{count}.jpg", gray[y:y+h, x:x+w])
        
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
        cv2.imshow('image', img)
    
    # Stop after 50 images
    if count >= 50:
        break
    
    # Press ESC to exit early
    if cv2.waitKey(1) == 27:
        break

print("Dataset collection complete")

cam.release()
cv2.destroyAllWindows()