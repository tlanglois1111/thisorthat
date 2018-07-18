# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from imutils.video import VideoStream
from threading import Thread
import numpy as np
import imutils
import time
import cv2

def light_tree(tree, sleep=5):
    # loop over all LEDs in the tree and randomly blink them with
    # varying intensities
    
    print ("got it!")

# define the paths to the Not Cats Keras deep learning model and
# audio file
MODEL_PATH = "cats_not_cats.model"

# initialize the total number of frames that *consecutively* contain
# cats along with threshold required to trigger the cats alarm
TOTAL_CONSEC = 0
TOTAL_THRESH = 20

# initialize is the cats alarm has been triggered
CATS = False

# load the model
print("[INFO] loading model...")
model = load_model(MODEL_PATH)

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
# vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)

# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=400)

    # prepare the image to be classified by our deep learning network
    image = cv2.resize(frame, (100, 100))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    # classify the input image and initialize the label and
    # probability of the prediction
    (cats, notCats) = model.predict(image)[0]
    label = "Not Cats"
    proba = notCats

    # check to see if cats was detected using our convolutional
    # neural network
    if cats > notCats:
        # update the label and prediction probability
        label = "Cats"
        proba = cats

        # increment the total number of consecutive frames that
        # contain cats
        TOTAL_CONSEC += 1

    # check to see if we should raise the cats alarm
    if not CATS and TOTAL_CONSEC >= TOTAL_THRESH:
        # indicate that cats has been found
        CATS = True

        # light up the christmas tree
        treeThread = Thread(target=light_tree, args=())
        treeThread.daemon = True
        treeThread.start()

    # otherwise, reset the total number of consecutive frames and the
    # cats alarm
    else:
        TOTAL_CONSEC = 0
        CATS = False

    # build the label and draw it on the frame
    label = "{}: {:.2f}%".format(label, proba * 100)
    frame = cv2.putText(frame, label, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
print("[INFO] cleaning up...")
cv2.destroyAllWindows()
vs.stop()
