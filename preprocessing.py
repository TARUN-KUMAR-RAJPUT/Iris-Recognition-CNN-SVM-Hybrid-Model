import cv2
import numpy as np
import glob
import pickle

detect_eye = cv2.CascadeClassifier('haarcascade_eye.xml')

numb_eye_two = 0

def my_image_transform(img,threshold):
    
    ans, threshold = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    open_info = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, identi_kernel_)
    close_info = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, identi_kernel_)

    open_close_info = cv2.bitwise_or(open_info, close_info, mask = None)

    return open_close_info,open_info,close_info
    

my_images = []
label = 0
my_img_vals = []
lables = []
detect_eye_list = []
iris_detect_eye_list=[]


for flpth in glob.iglob('DB/*'):
    count_=0
    
    for flpth_ in glob.iglob(flpth+'/L/*'):
        if flpth_[-1] == 'g':

            img	= cv2.imread(flpth_)
            img = cv2.resize(img,(200,150))

            img	= cv2.cvtColor(img,	cv2.COLOR_BGR2GRAY)
            
            my_images.append([img,count_,label,img])
            print(flpth_)
            count_ = count_+1

    for flpth_ in glob.iglob(flpth+'/R/*'):
        if flpth_[-1] == 'g':    
    
            img	= cv2.imread(flpth_)
            img = cv2.resize(img,(200,150))

            img	= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            my_images.append([img,count_,label,img])
            print(flpth_)
            count_ = count_+1
      
    label=label+1
        

print("count of images :",len(my_images))


numb_eye_ = 0
for i,j,L,c in my_images:

    i = cv2.resize(i,(400,300))

    eye_det_ = detect_eye.detectMultiScale(i, 1.01, 0)
    

    if len(eye_det_)>1:
        detect_eye_list.append(my_images[numb_eye_])
        print(numb_eye_)
        numb_eye_ = numb_eye_+1


print("count of eye detected = ",numb_eye_)
print("count of eye detected 2 = ",numb_eye_two)



numb_iris_ = 0

for i,j,L,c in detect_eye_list:

    sec_circle_ = cv2.HoughCircles(i, cv2.HOUGH_GRADIENT, 10, 100)

    if sec_circle_ is not None :
        
        sec_circle_ = np.round(sec_circle_[0, :]).astype("int")
        
        iris_detect_eye_list.append(my_images[numb_iris_])
        print(numb_iris_)
        numb_iris_ = numb_iris_+1

print("count of iris : = ",numb_iris_)

print("count of images : ",len(my_images))


my_images= iris_detect_eye_list



  
            
identi_kernel_ = np.ones((5,5),np.uint8)
import random

random.shuffle(my_images)

test=[]
for i,j,L,c in my_images:
    
    op_cl,op,cl = my_image_transform(i,0)
    op_cl_sum = sum(sum(op_cl))
    found = True
    for k in range(10,1000,10):
        
        working_img,open_info,close_info = my_image_transform(i,k)
        agrrgate_ = sum(sum(working_img))
        diffrence = agrrgate_- op_cl_sum
        
        if diffrence>800:
            found = False

            print("threshold  value of image = " ,k)            

            _, contours,_ = cv2.findContours(working_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

            for z in contours:

                x,y,w,h = cv2.boundingRect(z)
                if x+w<150 and y+h<200 and x-w//4>0:
                    
                    cv2.rectangle(working_img,(x,y),(x+w,y+h),(0,255,0),-2)
                    
            _, contours_2,_ = cv2.findContours(working_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            
            area_curr__max=0
            wth_max=0
            x_cordi=0
            y_cordi=0
            hght_max = 0
            for z in contours_2:
                x,y,w,h = cv2.boundingRect(z)
                new_area_curr_=h*w
                if x+w<150 and y+h<200 and new_area_curr_>area_curr__max and x-w//4>0:
                    area_curr__max = new_area_curr_
                    wth_max=w
                    x_cordi=x
                    y_cordi=y
                    hght_max = h
                    
                                        
            center_x = x_cordi+wth_max//2
            center_y = y_cordi+hght_max//2
            radius = 40

            if center_y-radius>0 and center_x-radius >0  and center_y+radius < 200 and center_x+radius < 150:
                new_roi = c[center_y-radius:center_y+radius, center_x-radius:center_x+radius]
                new_roi=cv2.resize(new_roi,(200,150))

                cv2.imwrite('processed_data/'+str(L)+'.'+str(j)+'.jpg',new_roi)

            else:
                center_y=c.shape[0]//2
                center_x=c.shape[1]//2
                new_roi = c[center_y-radius:center_y+radius, center_x-radius:center_x+radius]
                new_roi =cv2.resize(new_roi,(200,150))

                cv2.imwrite('processed_data/'+str(L)+'.'+str(j)+'.jpg',new_roi)

            test.append(i)
            my_img_vals.append(new_roi)
            lables.append(L)

    if  found :
        i = cv2.resize(i,(200,150))
    
        cv2.imwrite('processed_data/'+str(L)+'.'+str(j)+'.jpg',i)


print("final output length = ",len(my_img_vals))
print("Count of lables = ",len(lables))


