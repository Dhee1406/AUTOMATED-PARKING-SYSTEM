#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cv2 
import imutils
import pytesseract 
import pandas as pd
import time
import mysql.connector
import datetime
import sys
import re
import time
import requests


# In[2]:


from PyQt5 import QtCore, QtWidgets, uic
image_path = r'C:/Users/Dheeraj R/Downloads/MLproject/new/images/10.jpg'


# In[3]:


img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
img = imutils.resize(img, width=500)
cv2.imshow(image_path, img)


# In[4]:


gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# In[5]:


gray = cv2.bilateralFilter(gray, 11, 17, 17)


# In[6]:


edged = cv2.Canny(gray, 170, 200)


# In[7]:


cnts= cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
cnts=sorted(cnts, key = cv2.contourArea, reverse = True)[:30] 
NumberPlateCnt = None


# In[8]:


count = 0
for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:  
                NumberPlateCnt = approx 
                break


# In[9]:


mask = np.zeros(gray.shape,np.uint8)
new_image = cv2.drawContours(mask,[NumberPlateCnt],0,255,-1)
new_image = cv2.bitwise_and(img,img,mask=mask)
cv2.namedWindow("Final_image",cv2.WINDOW_NORMAL)
cv2.imshow("Final_image",new_image)


# In[10]:


config = ('-l eng --oem 1 --psm 3')
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
text = str(pytesseract.image_to_string(new_image, config=config))


# In[11]:


raw_data = {'date': [time.asctime( time.localtime(time.time()) )], 'v_number': [text]}


# In[12]:


df = pd.DataFrame(raw_data, columns = ['date', 'v_number'])
df.to_csv('data.csv')


# In[ ]:


print(text)
cv2.waitKey(0)


# In[ ]:




