import argparse
import base64
from datetime import datetime
import os
import shutil
import cv2
import numpy as np
import socketio
import eventlet
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO

# from keras.models import load_model
# import h5py
# from keras import __version__ as keras_version
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
from model_edge import *

sio = socketio.Server()
app = Flask(__name__)
model = CNN
prev_image_array = None


#transformations = transforms.Compose([transforms.Lambda(lambda x: (x / 255.0) - 0.5)])


class SimplePIController:
    def __init__(self, Kp, Ki):
        self.Kp = Kp
        self.Ki = Ki
        self.set_point = 0.
        self.error = 0.
        self.integral = 0.

    def set_desired(self, desired):
        self.set_point = desired

    def update(self, measurement):
        # proportional error
        self.error = self.set_point - measurement

        # integral error
        self.integral += self.error

        return self.Kp * self.error + self.Ki * self.integral


controller = SimplePIController(0.1, 0.002)
set_speed = 9
controller.set_desired(set_speed)


class filter_angle:
    def __init__(self,time_c=0.1):
        self.time_c=time_c
        self.angle=0
    def filter(self,steering_angle):
        self.angle=0
        self.angle=(1-self.time_c)*self.angle+self.time_c*steering_angle
        return self.angle

filter_1=filter_angle(0.1)


@sio.on('telemetry')
def telemetry(sid, data):
    
    if data: 
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #device = torch.device('cpu')
        # The current steering angle of the car
        steering_angle = float(data["steering_angle"])
        # The current throttle of the car
        throttle = float(data["throttle"])
        # The current speed of the car
        speed = float(data["speed"])
        # The current image from the center camera of the car
        imgString = data["image"]
        image = Image.open(BytesIO(base64.b64decode(imgString)))
        image =np.asarray(image)
        image = image[60:130,:]
        
        blur = cv2.GaussianBlur(image,(5,5),0)
        image = cv2.Canny(blur,150,250)

        
        image_array = np.asarray(image)
        #image = image_array[65:-25, :, :]
        #image = transformations(image)
        image=np.array(image)
        image=image*0.99/255 + 0.01
        
        
        #image = torch.Tensor(image)
        #print(image.shape)
        #image = image.view(1, 3, 70, 320)
        image=image.reshape(1,1,image.shape[-1],image.shape[-2])
        
        image = torch.from_numpy(image)
        image=image.float()
        image=image.to(device)
        image = Variable(image)
        #print(image.shape)
        
        #steering_angle = model(image).view(-1).data.numpy()[0]
        steering_angle = model(image)
        throttle = controller.update(float(speed))
        steering_angle=steering_angle.cpu()
        steering_angle=steering_angle.view(-1).data.numpy()
        #print("steering_angle=",sum(steering_angle), throttle)
        #print("steering_angle=",(steering_angle), throttle)
        #send_control(steering_angle, throttle)
       
        angle = filter_1.filter(sum(steering_angle))
        print("angle=",angle)
        send_control((angle),  throttle)
        # save frame
        if args.image_folder != '':
            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_filename = os.path.join(args.image_folder, timestamp)
            image.save('{}.jpg'.format(image_filename))
    else:
        # NOTE: DON'T EDIT THIS.
        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'model',
        type=str,
        help='Path to model h5 file. Model should be on the same path.'
    )
    parser.add_argument(
        'image_folder',
        type=str,
        nargs='?',
        default='',
        help='Path to image folder. This is where the images from the run will be saved.'
    )
    args = parser.parse_args()

    # check that model Keras version is same as local Keras version
    #checkpoint = torch.load((args.model), map_location=lambda storage, loc: storage)
    #model = checkpoint['model']
    checkpoint=torch.load((args.model))
    model = checkpoint['model']
    
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

    if args.image_folder != '':
        print("Creating image folder at {}".format(args.image_folder))
        if not os.path.exists(args.image_folder):
            os.makedirs(args.image_folder)
        else:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        print("RECORDING THIS RUN ...")
    else:
        print("NOT RECORDING THIS RUN ...")

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
