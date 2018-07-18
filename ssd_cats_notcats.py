# Import packages
from imutils.video.pivideostream import PiVideoStream

from keras.preprocessing.image import img_to_array
from keras.models import load_model

import os
import cv2
import numpy as np
from threading import Thread
import sys
import io
import socketserver
from threading import Condition
from http import server
import logging
import datetime
from random import randrange
import argparse
import time

# Set up camera constants
#IM_WIDTH = 1280
#IM_HEIGHT = 720
IM_WIDTH = 640    #Use smaller resolution for
IM_HEIGHT = 480   #slightly faster framerate
#IM_WIDTH = 300    #Use smaller resolution for
#IM_HEIGHT = 300   #slightly faster framerate

PAGE="""\
<html>
<head>
<title>picamera MJPEG streaming demo</title>
</head>
<body>
<h1>PiCamera MJPEG Streaming Demo</h1>
<img src="stream.mjpg" width="""+str(IM_WIDTH)+""" height="""+str(IM_HEIGHT)+""" />
</body>
</html>
"""

class StreamingOutput(object):
    def __init__(self):
        self.frame = None
        self.buffer = io.BytesIO()
        self.condition = Condition()

    def write(self, buf):
        if buf.startswith(b'\xff\xd8'):
            # New frame, copy the existing buffer's content and notify all
            # clients it's available
            self.buffer.truncate()
            with self.condition:
                self.frame = self.buffer.getvalue()
                self.condition.notify_all()
            self.buffer.seek(0)
        return self.buffer.write(buf)

class StreamingHandler(server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(301)
            self.send_header('Location', '/index.html')
            self.end_headers()
        elif self.path == '/index.html':
            content = PAGE.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.send_header('Content-Length', len(content))
            self.end_headers()
            self.wfile.write(content)
        elif self.path == '/stream.mjpg':
            self.send_response(200)
            self.send_header('Age', 0)
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
            self.end_headers()
            try:
                while True:
                    with output.condition:
                        output.condition.wait()
                        frame = output.frame
                    self.wfile.write(b'--FRAME\r\n')
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', len(frame))
                    self.end_headers()
                    self.wfile.write(frame)
                    self.wfile.write(b'\r\n')
            except Exception as e:
                logging.warning(
                    'Removed streaming client %s: %s',
                    self.client_address, str(e))
        else:
            self.send_error(404)
            self.end_headers()

class StreamingServer(socketserver.ThreadingMixIn, server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True


ap = argparse.ArgumentParser()
ap.add_argument("-d", "--debug", type=int, default=0,
                help="whether or not to log debug messages")
ap.add_argument("-i", "--inference", type=int, default=1,
                help="do inference on video stream")
ap.add_argument("-c", "--captureclass", type=int, default=0,
                help="capture the individual inferenced images")
ap.add_argument("-l", "--logfile", type=str,help="log to file")
args = vars(ap.parse_args())

doinfer = args['inference']
dodebug = args['debug']
capture_class_images = args['captureclass']

# configure logging
FORMAT = '%(asctime)-15s %(levelname)-8s %(message)s'
loglevel = logging.INFO
if (dodebug):
    loglevel = logging.DEBUG
if (args["logfile"]):
    logging.basicConfig(filename='./ssd.log', filemode='w',level=loglevel, format=FORMAT)
else:
    logging.basicConfig(level=loglevel, format=FORMAT)

# Select camera type (if user enters --usbcam when calling this script,
# a USB webcam will be used)
camera_type = 'picamera'

# This is needed since the working directory is the object_detection folder.
sys.path.append('..')

# Import utilites
#from utils import label_map_util
#from utils import visualization_utils as vis_util

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()
font = cv2.FONT_HERSHEY_SIMPLEX

# Initialize camera and perform object detection.
# The camera has to be set up and used differently depending on if it's a
# Picamera or USB webcam.

# I know this is ugly, but I basically copy+pasted the code for the object
# detection loop twice, and made one work for Picamera and the other work
# for USB.

ssd_img_height = 100
ssd_img_width = 100
normalize_coords = True
output_map = {0: 'background', 1: 'buddy', 2: 'jade', 3: 'lucy', 4: 'tim'}
# Set the threshold for detection
detection_threshold = 0.5
ssd_image_list = []
irand = randrange(0, 1000)

# initialize the total number of frames that *consecutively* contain
# cats along with threshold required to trigger the cats alarm
CAT_CONSEC = 0
NOTCAT_CONSEC = 0
CAT_THRESH = 4
NOTCAT_THRESH = 7200

### Picamera ###
if camera_type == 'picamera':

    vs = PiVideoStream(resolution=(IM_WIDTH, IM_HEIGHT)).start()

    output = StreamingOutput()

    try:

        today = datetime.datetime.now().strftime("%Y%m%d")

        ssd_counter = 0
        class_counter = 0
        address = ('', 10000)
        server = StreamingServer(address, StreamingHandler)

        logging.info("starting http server thread")
        thread = Thread(target=server.serve_forever)
        thread.daemon=True
        thread.start()
        logging.info("http server thread started")

        if (doinfer):
            logging.info("load model...")
            model_path = './cats_not_cats.model'

            model = load_model(model_path)

            logging.info("model loaded")

        time.sleep(2.0)

        # loop over some frames...this time using the threaded stream
        while True:
            # grab the frame from the threaded video stream
            frame = vs.read()
            #frame = imutils.resize(frame, width=400)

            t1 = cv2.getTickCount()

            image_saved = False

            if (doinfer):
                # Compute the scale in order to draw bounding boxes on the full resolution
                # image.
                yscale = float(frame.shape[0]/ssd_img_height)
                xscale = float(frame.shape[1]/ssd_img_width)

                frame_resize = cv2.resize(frame, (ssd_img_width, ssd_img_height))
                image = img_to_array(frame_resize)
                image = np.expand_dims(image, axis=0)

                (notCats, cats) = model.predict(image)[0]
                label = "Not Cat"
                proba = notCats

                #logging.debug('cats:{:03d}  notCats{:03d}', int(cats*100), int(notCats*100))

                if cats > notCats:
                    # update the label and prediction probability
                    label = "Cats"
                    proba = cats

                    # increment the total number of consecutive frames that
                    # contain cats
                    CAT_CONSEC += 1
                else:
                    NOTCAT_CONSEC += 1

                # check to see if we should raise the cats alarm
                if NOTCAT_CONSEC >= NOTCAT_THRESH:
                    logging.debug("write notcat image")

                    # light up the christmas tree
                    frame_filename = "{}_{:03d}_{}_{:03d}".format(today, ssd_counter, 'notcats', irand)
                    frame_path = "/tmp/cats/train/" + frame_filename + '.jpg'
                    cv2.imwrite(frame_path, frame)
                    ssd_counter += 1
                    NOTCAT_CONSEC = 0

                # check to see if we should raise the cats alarm
                if CAT_CONSEC >= CAT_THRESH:
                    logging.debug("write cat image")
                    # light up the christmas tree
                    frame_filename = "{}_{:03d}_{}_{:03d}".format(today, ssd_counter, 'cats', irand)
                    frame_path = "/tmp/cats/train/" + frame_filename + '.jpg'
                    cv2.imwrite(frame_path, frame)
                    ssd_counter += 1
                    image_saved = True;
                    CAT_CONSEC = 0

                # build the label and draw it on the frame
                label = "{}: {:.2f}%".format(label, proba * 100)
                frame = cv2.putText(frame, label, (425, 25),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if (cv2.getTickCount()%100 == 0 or frame_rate_calc < 2.0):
                fr = "FPS: {0:.2f}".format(frame_rate_calc)
            cv2.putText(frame,fr,(30,25),font,1,(255,255,0),1)

            #imgRGB = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            r, buf = cv2.imencode(".jpg",frame)
            output.write(bytearray(buf))

            # update the FPS counter
            t2 = cv2.getTickCount()
            time1 = (t2-t1)/freq
            frame_rate_calc = 1/time1

    finally:
        vs.stop()


