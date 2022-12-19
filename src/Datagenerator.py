
import cv2
import numpy as np
import os


class DataGenerator:
    def __init__(self,real_path,comic_path,batch_size,files,augment=False):
        #set bathsize
        self.batch_size=batch_size
        self.real_path=real_path
        self.comic_path=comic_path
        self.files=files  #list containing name of all the files
        self.augment=augment #boolean wether to augment or not
        #set pathsize
    def read_image(self,path):
        img=cv2.imread(path)
        img=cv2.resize(img,(128,128))#resize image to 128*128
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)#BGR to RGB
        return img
    def flip_transform(self,img):
        #Horizotal flip 
        return cv2.flip(img,1)
    
    def image_preprocessing(self,img):
        #scale the image down between 0 to 1
        img=img/255.0
        return img
    
    def __getitem__(self,start_index):
        
        data=[]
        labels=[]
        for indx in self.files[start_index:start_index+self.batch_size]:
            r_path=os.path.join(self.real_path,indx)
            c_path=os.path.join(self.comic_path,indx)
            real_img=self.image_preprocessing(self.read_image(r_path))
            comic_img=self.image_preprocessing(self.read_image(c_path))
            data.append(real_img)
            labels.append(comic_img)
            if self.augment==True:
                flip1=self.flip_transform(real_img)
                flip2=self.flip_transform(comic_img)
                data.append(flip1)
                labels.append(flip2)
        data=np.array(data)
        labels=np.array(labels)
        return labels,data
    
    