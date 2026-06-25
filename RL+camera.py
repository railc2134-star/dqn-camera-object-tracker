import torch
import torch.nn as nn
import numpy as np
import cv2
from collections import deque
import random
class envirmment():
    def __init__(self,lower_hsv,upper_hsv):
        self.cap=cv2.VideoCapture(0)
        self.lower_hsv=lower_hsv
        self.upper_hsv=upper_hsv
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    def get_object(self):
        self.ret,self.frame=self.cap.read()
        self.frame=cv2.cvtColor(self.frame,cv2.COLOR_BGR2HSV)
        self.mask=cv2.inRange(self.frame,self.lower_hsv,self.upper_hsv)
        self.contours, _ = cv2.findContours(self.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not self.contours:
            print("NO CONTOURS")
            return None,None,None,None
        c = max(self.contours, key=cv2.contourArea)
        if cv2.contourArea(c) < 500:
            print(f"BLOB TOO SMALL: {cv2.contourArea(c)}")
            return None,None,None,None
        
        M = cv2.moments(c)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        if n==1:
            print(f"OBJECT DETECTED at {cx},{cy}")
        return [cx/self.width , cy/self.height , self.crosshair_x/self.width,self.crosshair_y/self.width]
    def reset (self):
        self.crosshair_x=np.random.randint(0,self.width)
        self.crosshair_y=np.random.randint(0,self.height)
        stat=self.get_object()
        cx,cy=stat[0],stat[1]
        if cx is None:
            return None
        self.prev_distance=np.sqrt((cx-self.crosshair_x/self.width)**2 + (cy-self.crosshair_y/self.width)**2)
        return stat
    def step(self,action):
        if action==0:
            self.crosshair_y = min(self.height, self.crosshair_y+10)
        if action==1:
            self.crosshair_y = max(0, self.crosshair_y-10)
        if action==2:
            self.crosshair_x = min(self.width, self.crosshair_x+10)
        if action==3:
            self.crosshair_x = max(0, self.crosshair_x-10)
        cx, cy, _, _ = self.get_object()  # AFTER moving
        
        if cx is None:
            print(f"object not detected")
            return [-1, -1, self.crosshair_x/self.width, self.crosshair_y/self.width], -1, False
            
        distance=np.sqrt((cx-self.crosshair_x/self.width)**2 + (cy-self.crosshair_y/self.width)**2)
        if distance <0.05:
            done=True
            reword=1
        else:
            reword=self.prev_distance-distance
            done=False
        self.prev_distance = distance
        return [cx/self.width, cy/self.height, self.crosshair_x/self.width, self.crosshair_y/self.width], done, reword

class network(nn.Module):
    def __init__(self):
        super().__init__()
        self.input=nn.Linear(4,36)
        self.hidden=nn.Linear(36,64)
        self.hidden2=nn.Linear(64,64)
        self.output=nn.Linear(64,4)
    def forward(self,x):
        x=nn.functional.relu(self.input(x))
        x=nn.functional.relu(self.hidden(x))
        x=nn.functional.relu(self.hidden2(x))
        x=self.output(x)
        return x

class buffer:
    def __init__(self,capcity):
        self.buffer=deque(maxlen=capcity)
    def addd(self,state,reword,action,next_state,done):
        self.buffer.append((state,reword,action,next_state,done))
    def randomiser(self,batch_size):
        return random.sample(self.buffer,batch_size)
    def __len__(self):
        return len(self.buffer)
epsilon=1
epsilon_decay=0.995
epsilon_min=0.1
gamma=0.9
net=network()
target_net=network()
target_net.load_state_dict(net.state_dict())
capacity=10000
buffer_b=buffer(capacity)
buffer_batch=64
target_s=10
loss=nn.MSELoss()
optimize=torch.optim.Adam(net.parameters(),lr=0.001)
lower_hsv = (0, 70, 70)
losss = None
upper_hsv = (20, 255, 255)
env=envirmment(lower_hsv,upper_hsv)
train=False
if train==True:
    for episode in range(10000):
        done=False 
        steps=0
        n=1
        current_state = env.reset()
        if current_state is None or current_state[0] is None:
            continue
        while not done and steps<500:
            n=0
            steps+=1
            if epsilon > np.random.rand():
                action=np.random.randint(0,4)
            else:
                with torch.no_grad():
                    action=net(torch.FloatTensor(current_state)).argmax().item()
            next_state,done,reword=env.step(action)
            if next_state[2] == -1:
                break
            buffer_b.addd(current_state,reword,action,next_state,done)
            current_state=next_state
            if len(buffer_b) > buffer_batch:
                exp=buffer_b.randomiser(buffer_batch)   
                current_states,rewords,actions,next_states,dones=zip(*exp)
                current_state_b=torch.FloatTensor(current_states)
                reword_b=torch.FloatTensor(rewords)
                action_B=torch.LongTensor(actions)
                next_state_b=torch.FloatTensor(next_states)
                done_b=torch.FloatTensor([float(d) for d in dones])
                with torch.no_grad():
                    target=reword_b +target_net(next_state_b).max(1)[0]*gamma*(1-done_b)
                prediction=net(current_state_b).gather(1,action_B.unsqueeze(1)).squeeze(1)
                optimize.zero_grad()
                losss=loss(target,prediction)
                losss.backward()
                optimize.step()
        epsilon=max(epsilon*epsilon_decay,epsilon_min)
        if episode % 10==0:
            target_net.load_state_dict(net.state_dict())
        if episode % 1==0:
            print(f"episode = {episode} || steps ={steps} || loss = {losss} || action={action}|| epsilon = {epsilon}")
            print(f"saved succc")
n=0
if train==False:
    net.load_state_dict(torch.load('nett.pth'))
    net.eval()
    state = env.reset()
    print(state)
    if state is None or state[0] is None:
        print("No object detected")
    else:
        done = False
        while not done:
            with torch.no_grad():
                action = net(torch.FloatTensor(state)).argmax().item()
            state, done, reward = env.step(action)
            frame = env.frame.copy()
            frame = cv2.cvtColor(env.frame, cv2.COLOR_HSV2BGR)
            cv2.circle(frame, (env.crosshair_x, env.crosshair_y), 10, (0,255,0), 2)
            cv2.imshow('agent', frame)
            cv2.waitKey(1)
            print(f"action={action} | crosshair=({env.crosshair_x},{env.crosshair_y})")