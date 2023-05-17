from flask import Flask,render_template,Response
import cv2
import inference_code
import numpy as np

app=Flask(__name__)
# camera=cv2.VideoCapture('rtsp://admin:royal123@192.168.5.10:554')
# camera = cv2.VideoCapture('rtsp://admin:royal123@117.216.143.251:557/main')
camera = cv2.VideoCapture('output_554_1.mp4')

def generate_frames():
    counter = 0
    while True:
        success,frame=camera.read()
        if not success:
            break
        else:
            

            # encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90] # set JPEG quality (0-100)
            # result, jpeg_frame = cv2.imencode('.jpg', frame, encode_param)

            # # Convert the JPEG frame back to a numpy array
            # jpeg_frame_as_np = np.asarray(bytearray(jpeg_frame), dtype=np.uint8)

            # cv2.imwrite(f"frms/test{counter}.jpg",frame)
            # counter += 1

            frame = inference_code.show_inf(frame)


            ret,buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()

        yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')



@app.route('/video')
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=="__main__":
    app.run(host ='0.0.0.0',port=5000) 