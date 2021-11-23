import torch
from torchvision import models
import os
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset,DataLoader
import torch.nn as nn
import RPi.GPIO as GPIO
import time
import json 
from collections import OrderedDict
import requests


global a
num_class=4

model=models.mobilenet_v2(pretrained=True)
model.classifier=nn.Linear(1280,num_class)
model.load_state_dict(torch.load('/home/pi/Desktop/MobileNet_cpu.pt'))


trans_test = transforms.Compose([
      transforms.ToTensor(),
      transforms.Resize((224,224)),
      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class CustomDataSet(Dataset):

    def __init__(self, main_dir, transform):
        self.main_dir = main_dir
        self.transform = transform

        all_imgs = os.listdir(main_dir)
        self.total_imgs = sorted(all_imgs)

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        
        img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)

        return tensor_image


test_data = CustomDataSet('/home/pi/Desktop/image', transform=trans_test)
test_set = DataLoader(dataset = test_data, batch_size = 3)

result =[]

with torch.no_grad():
    result=[]
    for data in test_set:
        imgs = data
        pre_sum=0
        prediction=model(imgs)
        result.append(torch.argmax(prediction,1).tolist())
    a = result[0]   

def dis(distance):
    Buzz.start(50)
    Buzz.ChangeFrequency(523)
    time.sleep(0.5)
    Buzz.stop()
    time.sleep(0.3)

for i in range(3):
    # Initialize motor
    servo_pin0 = 21 #can
    servo_pin1 = 14 #paper
    servo_pin2 = 13 #plastic

    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM)
    

    #Initialize sensor
    #can
    triggerPin0 = 22
    echoPin0 = 23

    #paper
    triggerPin1 = 17
    echoPin1 = 27

    #plastic
    triggerPin2 = 20
    echoPin2 = 19

    pinPiezo = 24
    GPIO.setmode(GPIO.BCM)

    try:
      #can
        if a[i] == 0 or a[i] == 3:
            GPIO.setup(servo_pin0,GPIO.OUT)
            pwm0 = GPIO.PWM(servo_pin0,50)

            GPIO.setup(triggerPin0, GPIO.OUT)    
            GPIO.setup(echoPin0, GPIO.IN)
         
            GPIO.output(triggerPin0, GPIO.LOW)  
            time.sleep(0.00001) 
            GPIO.output(triggerPin0, GPIO.HIGH)

            while GPIO.input(echoPin0) == 0:  # Pulse occur
                start = time.time()
            while GPIO.input(echoPin0) == 1:  # Pulse return
                stop = time.time()

            rtTotime = stop - start                   # returntime = (end - start)

            distance = rtTotime * (34000 / 2 )
            Label = "CAN"
            print("distance [Can] : %.2f cm" %distance)     # Print distance
            time.sleep(0.2)

            GPIO.setup(pinPiezo, GPIO.OUT)
            Buzz = GPIO.PWM(pinPiezo, 400) 
            if distance <= 4:
                dis(distance)

            pwm0.start(3.0)
            pwm0.ChangeDutyCycle(3.0)
            time.sleep(10.0)
            pwm0.ChangeDutyCycle(12.5)
            time.sleep(1.0)
            pwm0.ChangeDutyCycle(0.0)
            pwm0.stop()

            GPIO.cleanup()

        
         
      # paper
        elif a[i] == 1:
            GPIO.setup(servo_pin1,GPIO.OUT)
            pwm1 = GPIO.PWM(servo_pin1,50)

           
            GPIO.setup(triggerPin1, GPIO.OUT)    
            GPIO.setup(echoPin1, GPIO.IN)
         
            GPIO.output(triggerPin1, GPIO.LOW)  
            time.sleep(0.00001) 
            GPIO.output(triggerPin1, GPIO.HIGH)

            while GPIO.input(echoPin1) == 0:  # Pulse occur
                start = time.time()
            while GPIO.input(echoPin1) == 1:  # Pulse return
                stop = time.time()

            rtTotime = stop - start                   # returntime = (end - start)

            distance = rtTotime * (34000 / 2 )
            Label = "PAPER"
            print("distance [Paper] : %.2f cm" %distance)     # Print distance
            time.sleep(0.2)

            GPIO.setup(pinPiezo, GPIO.OUT)
            Buzz = GPIO.PWM(pinPiezo, 400) 
            if distance <= 4:
                dis(distance)

            pwm1.start(3.0)
            pwm1.ChangeDutyCycle(3.0)
            time.sleep(10.0)
            pwm1.ChangeDutyCycle(12.5)
            time.sleep(1.0)
            pwm1.ChangeDutyCycle(0.0)
            pwm1.stop()

            GPIO.cleanup()
         
         
      # plastic
        elif a[i] == 2:
            GPIO.setup(servo_pin2,GPIO.OUT)
            pwm2 = GPIO.PWM(servo_pin2,50)

            GPIO.setup(triggerPin2, GPIO.OUT)    
            GPIO.setup(echoPin2, GPIO.IN)  

            GPIO.output(triggerPin2, GPIO.LOW)  
            time.sleep(0.00001) 
            GPIO.output(triggerPin2, GPIO.HIGH)
         
            while GPIO.input(echoPin2) == 0:  # Pulse occur
                start = time.time()
            while GPIO.input(echoPin2) == 1:  # Pulse return
                stop = time.time()

            rtTotime = stop - start                   # returntime = (end - start)

            distance = rtTotime * (34000 / 2 )
            Label = "Plastic"
            print("distance [Plastic] : %.2f cm" %distance)     # Print Distance
            time.sleep(0.2)

            GPIO.setup(pinPiezo, GPIO.OUT)
            Buzz = GPIO.PWM(pinPiezo, 400) 
            if distance <= 4:
                dis(distance)
         
            pwm2.start(3.0)
            pwm2.ChangeDutyCycle(3.0)
            time.sleep(10.0)
            pwm2.ChangeDutyCycle(12.5)
            time.sleep(1.0)
            pwm2.ChangeDutyCycle(0.0)
            pwm2.stop()
         
            GPIO.cleanup()

       # JSON
        Dis = round(((12 - distance)/12)*100)
        data = {
            'Label' : Label,
            'Volume' : Dis
        }
        json_data = json.dumps(data)
     
        #URL = "ec2"
        #res = requests.post(URL, data = json_data)
        print(json_data)

    except KeyboardInterrupt:
        GPIO.cleanup()