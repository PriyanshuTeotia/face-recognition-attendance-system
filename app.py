# from flask import Flask, render_template, Response
# import cv2

# app = Flask(__name__)

# # Load recognizer
# recognizer = cv2.face.LBPHFaceRecognizer_create()
# recognizer.read('trainer/trainer.yml')

# face_cascade = cv2.CascadeClassifier('cascade/haarcascade_frontalface_default.xml')

# names = ['Unknown', 'Priyanshu Teotia', 'Rahul']  # update as needed

# camera = cv2.VideoCapture(0)

# def generate_frames():
#     while True:
#         success, frame = camera.read()
#         if not success:
#             break
#         else:
#             gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#             faces = face_cascade.detectMultiScale(gray, 1.3, 5)

#             for (x, y, w, h) in faces:
#                 id, confidence = recognizer.predict(gray[y:y+h, x:x+w])

#                 if confidence < 60:
#                     name = names[id]
#                     confidence_text = f"{round(100 - confidence)}%"
#                 else:
#                     name = "Unknown"
#                     confidence_text = f"{round(100 - confidence)}%"

#                 cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
#                 cv2.putText(frame, name, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
#                 cv2.putText(frame, confidence_text, (x, y+h-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 1)

#             ret, buffer = cv2.imencode('.jpg', frame)
#             frame = buffer.tobytes()

#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# @app.route('/')
# def index():
#     return render_template('index.html')


# @app.route('/video')
# def video():
#     return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# if __name__ == "__main__":
#     app.run(debug=True)


from flask import Flask, render_template, Response, jsonify
import cv2

app = Flask(__name__)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')

face_cascade = cv2.CascadeClassifier('cascade/haarcascade_frontalface_default.xml')

names = ['Unknown', 'Priyanshu Teotia', 'Harsh Teotia']

camera = cv2.VideoCapture(0)

latest_result = {"name": "Waiting", "conf": 0}


def generate_frames():
    global latest_result

    while True:
        success, frame = camera.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            id, confidence = recognizer.predict(gray[y:y+h, x:x+w])

            if confidence < 60:
                name = names[id] if id < len(names) else "Unknown"
                conf = int(100 - confidence)
            else:
                name = "Unknown"
                conf = int(100 - confidence)

            latest_result = {"name": name, "conf": conf}

            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.putText(frame, name, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/result')
def result():
    return jsonify(latest_result)


if __name__ == "__main__":
    app.run(debug=True)