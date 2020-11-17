import os
import cv2 as cv

import face_recognition as fr

#taking the video files
video_f = cv.VideoCapture(os.path.abspath("videos/gangsof.mp4"))
length = int(video_f.get(cv.CAP_PROP_FRAME_COUNT)) # captuing the length of the file

# taking the image files into it
faizal_real = fr.load_image_file(os.path.abspath("images/faizal.png"))
asgar_real = fr.load_image_file(os.path.abspath("images/asgar.png"))
ehsaan_real = fr.load_image_file(os.path.abspath("images/ehsaan.png"))
mota_real = fr.load_image_file(os.path.abspath("images/mota.png"))
qamaar_real = fr.load_image_file(os.path.abspath("images/qamaar.png"))
sultan_real = fr.load_image_file(os.path.abspath("images/sultan.png"))

# now we have to encode the image files
faizal_enc = fr.face_encodings(faizal_real)[0]
asgar_enc = fr.face_encodings(asgar_real)[0]
ehsaan_enc = fr.face_encodings(ehsaan_real)[0]
mota_enc = fr.face_encodings(mota_real)[0]
qamaar_enc = fr.face_encodings(qamaar_real)[0]
sultan_enc = fr.face_encodings(sultan_real)[0]

faces = [faizal_enc,asgar_enc,ehsaan_enc,mota_enc,qamaar_enc,sultan_enc] #faces those are there after encoding

facial_number = 0
facial_points = []
face_enc = []
while True:
   value_present, frame = video_f.read()
   facial_number = facial_number + 1
   
   if not value_present:
       break
   rgb_frame = frame[:, :, ::-1]
   facial_points = fr.face_locations(rgb_frame, model = "cnn")
   facial_enc = fr.face_encodings(rgb_frame, facial_points)
   facial_names = []

   for enc in facial_enc:
       match = fr.compare_faces(faces, enc, tolerance = 0.5)
       name = ""
       if match[0]:
           name = "faizal khan"
       elif match[1]:
           name = "asgar khan"
       elif match[2]:
           name = "ehsaan qurashi"
       elif match[3]:
           name = "mota kasai"
       elif match[4]:
           name = "qamaar maqdoomi"
       elif match[5]:
           name = "sultan qurashi"
       else:
           name = "unknown"

       facial_names.append(name)


   for (top, right, bottom, left), name in zip(facial_points, facial_names):
       cv.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
       cv.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 255, 0), cv.FILLED)
       font = cv.FONT_HERSHEY_DUPLEX
       cv.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 2)

   codec = int(video_f.get(cv.CAP_PROP_FOURCC))
   fps = int(video_f.get(cv.CAP_PROP_FPS))
   frame_width = int(video_f.get(cv.CAP_PROP_FRAME_WIDTH))
   frame_height = int(video_f.get(cv.CAP_PROP_FRAME_HEIGHT))
   output_movie = cv.VideoWriter("output_{}.mp4".format(facial_number), codec, fps, (frame_width,frame_height))
   print("Writing frame {} / {}".format(facial_number, length))
   output_movie.write(frame)

video_f.release()
output_movie.release()
cv.destroyAllWindows()
