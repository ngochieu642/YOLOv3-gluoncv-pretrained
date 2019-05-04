import cv2
import time
from gluoncv import model_zoo, data, utils
from matplotlib import pyplot as plt
import gluoncv as gcv
import mxnet as mx
import numpy as np

capture = cv2.VideoCapture('../test/Survei_1.mp4')

#Delete the ctx if you are not using GPU
net = model_zoo.get_model('yolo3_darknet53_voc', pretrained = True, ctx=mx.gpu(0))

print('Net Structure:\n',net)
print('Net classes:\n',net.classes)
print("{0} classes".format(len(net.classes)))

def draw_box(img, bboxs, IDs, scores,RGB2BGR):
    """
    Receives an RGB image, bounding boxes , IDs and scores from the output of function net().
    Return an BGR image with rectangles and their corresponding (labels, scores)
    """
    #Change from RGB to BGR
    if(RGB2BGR):
        img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)

    #Chang to numpy array
    bboxs = bboxs.asnumpy()
    IDs   = IDs.asnumpy()
    scores= scores.asnumpy()

    #define rectange color
    myRecColor = (0,0,255)

    for (box,id,score) in zip(bboxs[0],IDs[0],scores[0]):

        #Get coordinate
        startX,startY = box[0].astype('int'), box[1].astype('int')
        endX,endY = box[2], box[3]

        #Get label and score
        name = net.classes[id[0].astype('int')]
        confidence = score[0]

        #Put labels and scores
        label = "{0} {1:.2f}".format(name,confidence)
        t_size = cv2.getTextSize(label,cv2.FONT_HERSHEY_PLAIN,1,1)[0]
        cv2.putText(img, label, (startX, startY - t_size[1] + 7), cv2.FONT_HERSHEY_SIMPLEX, .5, [0x3B, 0x52, 0xB1], 2)

        #Draw rectangle
        cv2.rectangle(img,(startX,startY),(endX,endY),myRecColor,1)

    return img

while(capture.isOpened()):
    stime = time.time()
    ret, frame = capture.read()

    if ret:
        # Image pre-processing
        frame = mx.nd.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).astype('uint8')

        rgb_nd, frame = gcv.data.transforms.presets.yolo.transform_test(frame, short=512, max_size=700)

        #Run detection, just use net(rgb_nd) if you are not using GPU
        class_IDs, scores, bounding_boxes = net(rgb_nd.as_in_context(mx.gpu(0)))

        #Draw things
        usedFrame = draw_box(frame,bounding_boxes,class_IDs,scores,1)

        #Calculate FPS
        FPS = 'FPS {:.1f}'.format(1/(time.time()-stime))
        usedFrameframe = cv2.putText(usedFrame,FPS,(100,100),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2)

        #Show result
        cv2.imshow('frame',usedFrame)

        #Escape key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        print('nothing to show')
        capture.release()
        cv2.destroyAllWindows()
        break
