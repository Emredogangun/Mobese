import tkinter as tk
from tkinter import *
from tkinter import messagebox
import cv2
import numpy as np
import pytesseract
import imutils
class Pencere(tk.Tk):
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
            
        def Arac_Plaka_Okuma_a():
            img = cv2.imread('‪D:\\OpenCV\\test_images\\licence_plate.jpg')
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            filtered = cv2.bilateralFilter(gray,6,250,250)
            edged = cv2.Canny(filtered,30,200)
            
            contours = cv2.findContours(edged,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(contours)
            cnts =sorted(cnts,key=cv2.contourArea,reverse=True)[:10]
            screen=None
            
            for c in cnts:
                epsilon = 0.018*cv2.arcLength(c,True)
                approx = cv2.approxPolyDP(c,epsilon,True)
                if len(approx)==4:
                    screen = approx
                    break
            
                
            mask = np.zeros(gray.shape,np.uint8)
            new_img = cv2.drawContours(mask,[screen],0,(255,255,255),-1)
            new_img = cv2.bitwise_and(img,img,mask = mask)
            
            (x,y) = np.where(mask == 255)
            (topx,topy) = (np.min(x),np.min(y))
            (bottomx,bottomy) = (np.max(x),np.max(y))
            cropped = gray[topx:bottomx+1,topy:bottomy+1]
            
            text = pytesseract.image_to_string(cropped,lang="eng")
            
            print("detected text:",text)
            var = messagebox.showinfo("detected text:",text)
            
        self.Video_Oku = tk.Button(text='Video Oku' ,width=25 ,height=4,command=Video_Oku_a)
        self.Video_Oku.pack(side=LEFT, fill=Y)
         
        self.Arac_Algila = tk.Button(text='Araç Algıla',width=25,height=4,command=Arac_Algila_a)
        self.Arac_Algila.pack(side=TOP, fill=X)
        
        self.Arac_Sayma = tk.Button(text='Araç Sayma',width=25,height=2,command=Arac_Sayma_a)
        self.Arac_Sayma.pack(side=TOP, fill=X)
        
        self.Arac_Sayma_TekYon = tk.Button(text='Araç Sayma TekYon',width=25,height=2,command=Arac_Sayma_TekYon_a)
        self.Arac_Sayma_TekYon.pack(side=TOP, fill=X)
        
        self.Arac_Plaka_Okuma = tk.Button(text='Araç Plaka Okuma',width=25,height=4,command=Arac_Plaka_Okuma_a)
        self.Arac_Plaka_Okuma.pack(side=TOP, fill=X)
        
        self.Arac_Hiz_Tespit = tk.Button(text='Araç Hız Tespit',width=25,height=4,command=dialog)
        self.Arac_Hiz_Tespit.pack(side=TOP, fill=X)
                
        self.Arac_Yogunlugu = tk.Button(text='Araç Yoğunluğu',width=25,height=4,command=dialog)
        self.Arac_Yogunlugu.pack(side=TOP, fill=X)
        
           
        self.düğme = tk.Button(text='Çık',width=25,height=3, command=self.çıkış)
        self.düğme.pack()

    def çıkış(self):
        self.etiket['text'] = 'Elveda  Mobose Sistemi'
        self.düğme['text'] = 'Bekleyin...'
        self.düğme['state'] = 'disabled'
        self.after(1000, self.destroy)

pencere = Pencere()
pencere.mainloop()