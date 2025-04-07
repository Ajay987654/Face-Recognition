import cv2

face_cap = cv2.CascadeClassifier("c:/Users/ME/AppData/Local/Packages/PythonSoftwareFoundation.Python.3.10_qbz5n2kfra8p0/LocalCache/local-packages/Python310/site-packages/cv2/data/haarcascade_frontalface_default.xml")
video_cap = cv2.VideoCapture(0)

while True:
    ret, video = video_cap.read()
    col = cv2.cvtColor(video,cv2.COLOR_BGR2GRAY)
    faces = face_cap.detectMultiScale(
      col,
      scaleFactor = 1.1,
      minNeighbors=5,
      minSize=(30,30),
      flags=cv2.CASCADE_SCALE_IMAGE
    )
    for (x,y,w,h) in faces:
      cv2.rectangle(video,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.imshow("Video_Live", video)  # Don't forget to display the frame

    if cv2.waitKey(10) == ord("a"):
        break

video_cap.release()
cv2.destroyAllWindows()
