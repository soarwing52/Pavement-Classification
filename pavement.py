from tkinter import *
from tkinter import filedialog
import os
import tensorflow as tf
import numpy as np
import cv2
import threading

class Recognition:
    def __init__(self, root):
        frame = Frame(root, bd=5)
        frame.pack()
        s_var = StringVar()
        l = Label(frame, textvariable=s_var, bg='white', bd=5, width=40)
        b = Button(frame, text='select folder', height=2, command=lambda: self.get_dir(s_var))
        b.pack()
        l.pack()

        frame_2 = Frame(root, height=20, bd=5)
        frame_2.pack()
        Button(frame_2, text='recognize pavement', command=self.open_thread).grid(row=1, column=1)

        frame_b = Frame(root)
        frame_b.pack()
        Label(frame_b,text='current image',bd=5,width=80).pack()
        self.current = StringVar()
        Label(frame_b,textvariable=self.current,bd=5,bg='white',width=80).pack()
        Label(frame_b, text='type', bd=5, width=80).pack()
        self.predict = StringVar()
        Label(frame_b,textvariable=self.predict,bd=5,bg='white',width=50).pack()


    def get_dir(self, var):
        self.dir_name = filedialog.askdirectory()
        var.set(self.dir_name)

    def open_thread(self):
        T1 = threading.Thread(target=self.tensor_recog,name='T1')
        T1.start()

    def tensor_recog(self):
        TEST_IMAGES_DIR = self.dir_name
        RETRAINED_LABELS_TXT_FILE_LOC = os.getcwd() + "/" + "retrained_labels.txt"
        RETRAINED_GRAPH_PB_FILE_LOC = os.getcwd() + "/" + "retrained_graph.pb"

        SCALAR_RED = (0.0, 0.0, 255.0)
        SCALAR_BLUE = (255.0, 0.0, 0.0)

        # get a list of classifications from the labels file
        classifications = []
        # for each line in the label file . . .
        for currentLine in tf.gfile.GFile(RETRAINED_LABELS_TXT_FILE_LOC):
            # remove the carriage return
            classification = currentLine.rstrip()
            # and append to the list
            classifications.append(classification)
        # end for

        # show the classifications to prove out that we were able to read the label file successfully
        print("classifications = " + str(classifications))

        # load the graph from file
        with tf.gfile.FastGFile(RETRAINED_GRAPH_PB_FILE_LOC, 'rb') as retrainedGraphFile:
            # instantiate a GraphDef object
            graphDef = tf.GraphDef()
            # read in retrained graph into the GraphDef object
            graphDef.ParseFromString(retrainedGraphFile.read())
            # import the graph into the current default Graph, note that we don't need to be concerned with the return value
            _ = tf.import_graph_def(graphDef, name='')
        # end with

        with tf.Session() as sess:
            # for each file in the test images directory . . .
            for root, dirs, images in os.walk(TEST_IMAGES_DIR):
                for img in images:
                    fileName = os.path.join(root, img)
                    self.current.set(fileName)
                    # if the file does not end in .jpg or .jpeg (case-insensitive), continue with the next iteration of the for loop
                    if not (fileName.lower().endswith(".jpg") or fileName.lower().endswith(".jpeg")):
                        print(fileName)
                        continue
                    # end if

                    # show the file name on std out
                    print(fileName)

                    # get the file name and full path of the current image file
                    imageFileWithPath = os.path.join(TEST_IMAGES_DIR, fileName)
                    # attempt to open the image with OpenCV
                    openCVImage = cv2.imread(imageFileWithPath)
                    # if we were not able to successfully open the image, continue with the next iteration of the for loop
                    if openCVImage is None:
                        print("unable to open " + fileName + " as an OpenCV image")
                        continue
                    # end if
                    height, width, channel = openCVImage.shape
                    openCVImage = openCVImage[round(height / 3): height,
                                  round(0.5 * width - 0.3 * width): round(0.5 * width + 0.3 * width)]
                    # get the final tensor from the graph
                    finalTensor = sess.graph.get_tensor_by_name('final_result:0')

                    # convert the OpenCV image (numpy array) to a TensorFlow image
                    tfImage = np.array(openCVImage)[:, :, 0:3]

                    # run the network to get the predictions
                    predictions = sess.run(finalTensor, {'DecodeJpeg:0': tfImage})

                    # sort predictions from most confidence to least confidence
                    sortedPredictions = predictions[0].argsort()[-len(predictions[0]):][::-1]

                    print("---------------------------------------")

                    # keep track of if we're going through the next for loop for the first time so we can show more info about
                    # the first prediction, which is the most likely prediction (they were sorted descending above)
                    onMostLikelyPrediction = True
                    # for each prediction . . .
                    for prediction in sortedPredictions:
                        strClassification = classifications[prediction]

                        # if the classification (obtained from the directory name) ends with the letter "s", remove the "s" to change from plural to singular
                        if strClassification.endswith("s"):
                            strClassification = strClassification[:-1]
                        # end if

                        # get confidence, then get confidence rounded to 2 places after the decimal
                        confidence = predictions[0][prediction]

                        # if we're on the first (most likely) prediction, state what the object appears to be and show a % confidence to two decimal places
                        if onMostLikelyPrediction:
                            # get the score as a %
                            scoreAsAPercent = confidence * 100.0
                            # show the result to std out
                            result ="the object appears to be a " + strClassification + ", " + "{0:.2f}".format(
                                scoreAsAPercent) + "% confidence"
                            print(result)
                            # mark that we've show the most likely prediction at this point so the additional information in
                            # this if statement does not show again for this image
                            onMostLikelyPrediction = False
                            with open('predict_result.txt', 'a') as logfile:
                                msg = '{},{},{} \n'.format(fileName, strClassification, scoreAsAPercent)
                                self.predict.set(result)
                                logfile.write(msg)
                        # end if

                        # for any prediction, show the confidence as a ratio to five decimal places
                        print(strClassification + " (" + "{0:.5f}".format(confidence) + ")")
                    # end for


def main():
    root = Tk()
    root.title('image blocker')
    root.geometry('600x250')
    Recognition(root)
    root.mainloop()


if __name__ == '__main__':
    main()