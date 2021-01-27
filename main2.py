import tkinter as tk
from tkinter import *
from tkinter import messagebox
import cv2
import numpy as np
import speed_check
import dlib
import time
import threading
import math
import imutils
import Main
class aa2(tk.Tk):
    def __init__(self):
        super().__init__()
        self.protocol('WM_DELETE_WINDOW', self.çıkış)
        self.geometry("500x500")
        self.etiket = tk.Label(text='Merhaba Mobose Sistemi')
        self.etiket.pack()
        
        def dialog():
            var = messagebox.showinfo("Uyarı" , "asd")

        def Video_Oku_a():
            vid = cv2.VideoCapture('D:\\OpenCV\\test_videos\\car.mp4')
            car_cascade = cv2.CascadeClassifier('D:\\OpenCV\\car_cascade\\car_cascade.xml')
            while True:
                ret,frame = vid.read()
                frame = cv2.resize(frame,(480,480))
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                cars = car_cascade.detectMultiScale(gray,1.3,3)
            
                cv2.imshow('Video Okuma',frame)
            
                if cv2.waitKey(5) & 0xFF == ord('q'):
                    break
                
            vid.release()
            cv2.destroyAllWindows()
            
        def Arac__Plaka_Algila_a():
            # Read the image file
            image = cv2.imread('D:\\OpenCV\\test_images\\licence_plate.jpg')

            # Resize the image - change width to 500
            image = imutils.resize(image, width=500)

            # Display the original image
            cv2.imshow("Original Image", image)

            # RGB to Gray scale conversion
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            cv2.imshow("1 - Grayscale Conversion", gray)

            # Noise removal with iterative bilateral filter(removes noise while preserving edges)
            gray = cv2.bilateralFilter(gray, 11, 17, 17)
            cv2.imshow("2 - Bilateral Filter", gray)

            # Find Edges of the grayscale image
            edged = cv2.Canny(gray, 170, 200)
            cv2.imshow("4 - Canny Edges", edged)

            # Find contours based on Edges
            (new, cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            cnts=sorted(cnts, key = cv2.contourArea, reverse = True)[:30] #sort contours based on their area keeping minimum required area as '30' (anything smaller than this will not be considered)
            NumberPlateCnt = None #we currently have no Number plate contour

            # loop over our contours to find the best possible approximate contour of number plate
            count = 0
            for c in cnts:
                    peri = cv2.arcLength(c, True)
                    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                    if len(approx) == 4:  # Select the contour with 4 corners
                        NumberPlateCnt = approx #This is our approx Number Plate Contour
                        break


            # Drawing the selected contour on the original image
            cv2.drawContours(image, [NumberPlateCnt], -1, (0,255,0), 3)
            cv2.imshow("Final Image With Number Plate Detected", image)

            cv2.waitKey(0) #Wait for user input before closing the images displayed
        def Arac__Plaka_Okuma_a():
           Main.main();
            
        def Arac_Algila_a():
            vid = cv2.VideoCapture('D:\\OpenCV\\test_videos\\car.mp4')
            #car_cascade = cv2.CascadeClassifier('D:\\OpenCV\\haarCascades\\car.xml')
            car_cascade = cv2.CascadeClassifier('D:\\OpenCV\\car_cascade\\car_cascade.xml')
            
            
            while True:
                ret,frame = vid.read()
                frame = cv2.resize(frame,(640,480))
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                cars = car_cascade.detectMultiScale(gray,1.3,3)
            
                for (x,y,w,h) in cars:
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),3)
                
                cv2.imshow('video',frame)
            
                if cv2.waitKey(5) & 0xFF == ord('q'):
                    break
                
            vid.release()
            cv2.destroyAllWindows()

        def Arac_Hiz_Tespit_a():
           
            carCascade = cv2.CascadeClassifier('myhaar.xml')
            video = cv2.VideoCapture('cars.mp4')

            WIDTH = 1280
            HEIGHT = 720


            def estimateSpeed(location1, location2):
                    d_pixels = math.sqrt(math.pow(location2[0] - location1[0], 2) + math.pow(location2[1] - location1[1], 2))
                    # ppm = location2[2] / carWidht
                    ppm = 8.8
                    d_meters = d_pixels / ppm
                    #print("d_pixels=" + str(d_pixels), "d_meters=" + str(d_meters))
                    fps = 18
                    speed = d_meters * fps * 3.6
                    return speed
                    

            def trackMultipleObjects():
                    rectangleColor = (0, 255, 0)
                    frameCounter = 0
                    currentCarID = 0
                    fps = 0
                    
                    carTracker = {}
                    carNumbers = {}
                    carLocation1 = {}
                    carLocation2 = {}
                    speed = [None] * 1000
                    
                    # Write output to video file
                    out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (WIDTH,HEIGHT))


                    while True:
                            start_time = time.time()
                            rc, image = video.read()
                            if type(image) == type(None):
                                    break
                            
                            image = cv2.resize(image, (WIDTH, HEIGHT))
                            resultImage = image.copy()
                            
                            frameCounter = frameCounter + 1
                            
                            carIDtoDelete = []

                            for carID in carTracker.keys():
                                    trackingQuality = carTracker[carID].update(image)
                                    
                                    if trackingQuality < 7:
                                            carIDtoDelete.append(carID)
                                            
                            for carID in carIDtoDelete:
                                    print ('Removing carID ' + str(carID) + ' from list of trackers.')
                                    print ('Removing carID ' + str(carID) + ' previous location.')
                                    print ('Removing carID ' + str(carID) + ' current location.')
                                    carTracker.pop(carID, None)
                                    carLocation1.pop(carID, None)
                                    carLocation2.pop(carID, None)
                            
                            if not (frameCounter % 10):
                                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                                    cars = carCascade.detectMultiScale(gray, 1.1, 13, 18, (24, 24))
                                    
                                    for (_x, _y, _w, _h) in cars:
                                            x = int(_x)
                                            y = int(_y)
                                            w = int(_w)
                                            h = int(_h)
                                    
                                            x_bar = x + 0.5 * w
                                            y_bar = y + 0.5 * h
                                            
                                            matchCarID = None
                                    
                                            for carID in carTracker.keys():
                                                    trackedPosition = carTracker[carID].get_position()
                                                    
                                                    t_x = int(trackedPosition.left())
                                                    t_y = int(trackedPosition.top())
                                                    t_w = int(trackedPosition.width())
                                                    t_h = int(trackedPosition.height())
                                                    
                                                    t_x_bar = t_x + 0.5 * t_w
                                                    t_y_bar = t_y + 0.5 * t_h
                                            
                                                    if ((t_x <= x_bar <= (t_x + t_w)) and (t_y <= y_bar <= (t_y + t_h)) and (x <= t_x_bar <= (x + w)) and (y <= t_y_bar <= (y + h))):
                                                            matchCarID = carID
                                            
                                            if matchCarID is None:
                                                    print ('Creating new tracker ' + str(currentCarID))
                                                    
                                                    tracker = dlib.correlation_tracker()
                                                    tracker.start_track(image, dlib.rectangle(x, y, x + w, y + h))
                                                    
                                                    carTracker[currentCarID] = tracker
                                                    carLocation1[currentCarID] = [x, y, w, h]

                                                    currentCarID = currentCarID + 1
                            
                            #cv2.line(resultImage,(0,480),(1280,480),(255,0,0),5)


                            for carID in carTracker.keys():
                                    trackedPosition = carTracker[carID].get_position()
                                                    
                                    t_x = int(trackedPosition.left())
                                    t_y = int(trackedPosition.top())
                                    t_w = int(trackedPosition.width())
                                    t_h = int(trackedPosition.height())
                                    
                                    cv2.rectangle(resultImage, (t_x, t_y), (t_x + t_w, t_y + t_h), rectangleColor, 4)
                                    
                                    # speed estimation
                                    carLocation2[carID] = [t_x, t_y, t_w, t_h]
                            
                            end_time = time.time()
                            
                            if not (end_time == start_time):
                                    fps = 1.0/(end_time - start_time)
                            
                            #cv2.putText(resultImage, 'FPS: ' + str(int(fps)), (620, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)


                            for i in carLocation1.keys():	
                                    if frameCounter % 1 == 0:
                                            [x1, y1, w1, h1] = carLocation1[i]
                                            [x2, y2, w2, h2] = carLocation2[i]
                            
                                            # print 'previous location: ' + str(carLocation1[i]) + ', current location: ' + str(carLocation2[i])
                                            carLocation1[i] = [x2, y2, w2, h2]
                            
                                            # print 'new previous location: ' + str(carLocation1[i])
                                            if [x1, y1, w1, h1] != [x2, y2, w2, h2]:
                                                    if (speed[i] == None or speed[i] == 0) and y1 >= 275 and y1 <= 285:
                                                            speed[i] = estimateSpeed([x1, y1, w1, h1], [x2, y2, w2, h2])

                                                    #if y1 > 275 and y1 < 285:
                                                    if speed[i] != None and y1 >= 180:
                                                            cv2.putText(resultImage, str(int(speed[i])) + " km/hr", (int(x1 + w1/2), int(y1-5)),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
                                                    
                                                    #print ('CarID ' + str(i) + ': speed is ' + str("%.2f" % round(speed[i], 0)) + ' km/h.\n')

                                                    #else:
                                                    #	cv2.putText(resultImage, "Far Object", (int(x1 + w1/2), int(y1)),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                                                            #print ('CarID ' + str(i) + ' Location1: ' + str(carLocation1[i]) + ' Location2: ' + str(carLocation2[i]) + ' speed is ' + str("%.2f" % round(speed[i], 0)) + ' km/h.\n')
                            cv2.imshow('result', resultImage)
                            # Write the frame into the file 'output.avi'
                            #out.write(resultImage)


                            if cv2.waitKey(33) == 27:
                                    break
                    
                    cv2.destroyAllWindows()
            if __name__ == '__main__':
                trackMultipleObjects()   
                        
        def Arac_Yogunlugu_a():
            #vid = cv2.VideoCapture("D:\\OpenCV\\test_videos\\traffic.avi")
            vid = cv2.VideoCapture('D:\\OpenCV\\test_videos\\car.mp4')
            backsub = cv2.createBackgroundSubtractorMOG2()
            car_cascade = cv2.CascadeClassifier('D:\\OpenCV\\car_cascade\\car_cascade.xml')
            c = 0
            
            while True:
                ret,frame = vid.read()
                frame = cv2.resize(frame,(640,480))
                if ret:
                    fgmask = backsub.apply(frame)
                    cv2.line(frame,(0,400),(640,400),(0,255,0),2)
                    cv2.line(frame,(0,420),(640,420),(0,255,0),2)
            
                    contours,hierarchy = cv2.findContours(fgmask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                    try : hierarchy = hierarchy[0]
                    except: hierarchy=[]
            
                    for contour,hier in zip(contours,hierarchy):
                        (x,y,w,h) = cv2.boundingRect(contour)
                        if w>30 and h >30 and w<100 and h<100:
                            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),3)
                            if x>50 and x<70 and w<100 and h<100:
                                c+=1
            
                    # cv2.putText(source_image,text,coordinates,font,size,color,thickness,better look)          
                    if c<3:
                        cv2.putText(frame,"Y: "+"Az",(0,100),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),2,cv2.LINE_AA)
                    if c==3:
                        cv2.putText(frame,"Y: "+"Normal",(0,100),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),2,cv2.LINE_AA)
                    if c>3:
                        cv2.putText(frame,"Y: "+"Cok",(0,100),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),2,cv2.LINE_AA)

                    
            
                    cv2.imshow("Araba Sayımı",frame)
                    cv2.imshow("fgmask",fgmask)
                    
                    if cv2.waitKey(40) & 0xFF==ord('q'):
                        break
            
            vid.release()
            cv2.destroyAllWindows()   
            
        def Arac_Sayma_a():
            #vid = cv2.VideoCapture("D:\\OpenCV\\test_videos\\traffic.avi")
            vid = cv2.VideoCapture('D:\\OpenCV\\test_videos\\car.mp4')
            backsub = cv2.createBackgroundSubtractorMOG2()
            car_cascade = cv2.CascadeClassifier('D:\\OpenCV\\car_cascade\\car_cascade.xml')
            c = 0
            
            while True:
                ret,frame = vid.read()
                frame = cv2.resize(frame,(640,480))
                if ret:
                    fgmask = backsub.apply(frame)
                    cv2.line(frame,(0,400),(640,400),(0,255,0),2)
                    cv2.line(frame,(0,420),(640,420),(0,255,0),2)
            
                    contours,hierarchy = cv2.findContours(fgmask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                    try : hierarchy = hierarchy[0]
                    except: hierarchy=[]
            
                    for contour,hier in zip(contours,hierarchy):
                        (x,y,w,h) = cv2.boundingRect(contour)
                        if w>30 and h >30 and w<100 and h<100:
                            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),3)
                            if x>50 and x<70 and w<100 and h<100:
                                c+=1
            
                    # cv2.putText(source_image,text,coordinates,font,size,color,thickness,better look)          
                    cv2.putText(frame,"Araba: "+str(c),(90,100),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),2,cv2.LINE_AA)
                    
            
                    cv2.imshow("Araba Sayımı",frame)
                    cv2.imshow("fgmask",fgmask)
                    
                    if cv2.waitKey(40) & 0xFF==ord('q'):
                        break
            
            vid.release()
            cv2.destroyAllWindows()    
        
            
        def Arac_Yogunlugu_TekYon_a():
            vid = cv2.VideoCapture("D:\\OpenCV\\test_videos\\traffic.avi")
            backsub = cv2.createBackgroundSubtractorMOG2()
            c = 0
            
            while True:
                ret,frame = vid.read()
                if ret:
                    fgmask = backsub.apply(frame)
                    cv2.line(frame,(50,0),(50,300),(0,255,0),2)
                    cv2.line(frame,(70,0),(70,300),(0,255,0),2)
            
                    contours,hierarchy = cv2.findContours(fgmask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                    try : hierarchy = hierarchy[0]
                    except: hierarchy=[]
            
                    for contour,hier in zip(contours,hierarchy):
                        (x,y,w,h) = cv2.boundingRect(contour)
                        if w>40 and h >40:
                            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),3)
                            if x>50 and x<70:
                                c+=1
            
                    # cv2.putText(source_image,text,coordinates,font,size,color,thickness,better look)      
                    if c<3:
                        cv2.putText(frame,"Y: "+"Az",(0,100),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),2,cv2.LINE_AA)
                    if c==3:
                        cv2.putText(frame,"Y: "+"Normal",(0,100),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),2,cv2.LINE_AA)
                    if c>3:
                        cv2.putText(frame,"Y: "+"Cok",(0,100),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),2,cv2.LINE_AA)

                    cv2.imshow("Car Counting",frame)
                    cv2.imshow("fgmask",fgmask)
                    
                    if cv2.waitKey(40) & 0xFF==ord('q'):
                        break
            
            vid.release()
            cv2.destroyAllWindows()    
        def Arac_Sayma_TekYon_a():
            vid = cv2.VideoCapture("D:\\OpenCV\\test_videos\\traffic.avi")
            backsub = cv2.createBackgroundSubtractorMOG2()
            c = 0
            
            while True:
                ret,frame = vid.read()
                if ret:
                    fgmask = backsub.apply(frame)
                    cv2.line(frame,(50,0),(50,300),(0,255,0),2)
                    cv2.line(frame,(70,0),(70,300),(0,255,0),2)
            
                    contours,hierarchy = cv2.findContours(fgmask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                    try : hierarchy = hierarchy[0]
                    except: hierarchy=[]
            
                    for contour,hier in zip(contours,hierarchy):
                        (x,y,w,h) = cv2.boundingRect(contour)
                        if w>40 and h >40:
                            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),3)
                            if x>50 and x<70:
                                c+=1
            
                    # cv2.putText(source_image,text,coordinates,font,size,color,thickness,better look)          
                    cv2.putText(frame,"car: "+str(c),(90,100),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),2,cv2.LINE_AA)
                    
            
                    cv2.imshow("Car Counting",frame)
                    cv2.imshow("fgmask",fgmask)
                    
                    if cv2.waitKey(40) & 0xFF==ord('q'):
                        break
            
            vid.release()
            cv2.destroyAllWindows()     
            
        
            
        self.Video_Oku = tk.Button(text='Video Oku' ,width=25 ,height=4,command=Video_Oku_a)
        self.Video_Oku.pack(side=LEFT, fill=Y)
         
        self.Arac_Algila = tk.Button(text='Araç Algıla',width=25,height=4,command=Arac_Algila_a)
        self.Arac_Algila.pack(side=TOP, fill=X)
        
        self.Arac_Sayma = tk.Button(text='Araç Sayma',width=25,height=2,command=Arac_Sayma_a)
        self.Arac_Sayma.pack(side=TOP, fill=X)
        
        self.Arac_Sayma_TekYon = tk.Button(text='Araç Sayma TekYon',width=25,height=2,command=Arac_Sayma_TekYon_a)
        self.Arac_Sayma_TekYon.pack(side=TOP, fill=X)
        
        self.Arac_Plaka_Okuma = tk.Button(text='Araç Plaka Algıla',width=25,height=2,command=Arac__Plaka_Algila_a)
        self.Arac_Plaka_Okuma.pack(side=TOP, fill=X)

        self.Arac_Plaka_Okuma = tk.Button(text='Araç Plaka Okuma',width=25,height=2,command=Arac__Plaka_Okuma_a)
        self.Arac_Plaka_Okuma.pack(side=TOP, fill=X)
        
        self.Arac_Hiz_Tespit = tk.Button(text='Araç Hız Tespit',width=25,height=4,command=Arac_Hiz_Tespit_a)
        self.Arac_Hiz_Tespit.pack(side=TOP, fill=X)
        
        self.Arac_Yogunlugu = tk.Button(text='Araç Yoğunluğu',width=25,height=2,command=Arac_Yogunlugu_a)
        self.Arac_Yogunlugu.pack(side=TOP, fill=X)
        
        self.Arac_Yogunlugu = tk.Button(text='Araç YoğunluğuTekYon',width=25,height=2,command=Arac_Yogunlugu_TekYon_a)
        self.Arac_Yogunlugu.pack(side=TOP, fill=X)
        
           
        self.düğme = tk.Button(text='Çık',width=25,height=3, command=self.çıkış)
        self.düğme.pack()

    def çıkış(self):
        self.etiket['text'] = 'Elveda  Mobose Sistemi'
        self.düğme['text'] = 'Bekleyin...'
        self.düğme['state'] = 'disabled'
        self.after(1000, self.destroy)

aa2 = aa2()
aa2.mainloop()
