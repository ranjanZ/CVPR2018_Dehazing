#same as main buy process file to  file

import numpy as np
from scipy import misc
from skimage.color import rgb2gray
import sys
import os
import skimage.transform  as trans
from gf import *
from skimage.io import imread,imsave
import time
from model import *
import matplotlib.pyplot as plt


#if you want to use only CPU thn 
#os.environ['CUDA_VISIBLE_DEVICES'] ='-1'

# run main_file.py  ../data/cvpr2018/IndoorValidationHazy/*.png      
#save the file in the same directory with out2 extension


#put your base directory here
base_dir="/home/ranjan/dehazing/server/CVPR2018_Dehazing/"


### Initialization 
b_patch_X = 128
b_patch_Y = 128
####


try:
    hazy_image_path=sys.argv[1]
    f_out="out2_"+hazy_image_path.split("/")[-1]
    out_image_path= hazy_image_path
except:
    print"-----------------------------------------------------------------------------------------------------"
    print "Please provide hazy image directory  as 1st argument and  Outputdirectory as second argument"
    print "running in default mode....."
    print"-----------------------------------------------------------------------------------------------------"
    hazy_image_dir= "../data/hazy_img/"
    out_image_dir= "../data/out/"





def read_image(path):
        image = misc.imread(path,mode="RGB")
        if(len(image.shape)!=3 and image.shape[-1]!=0):
            print"ERROR: Please provide RGB Image "
            exit()
        image = image.astype('float32')/255
        return image



def f(image,L1,L2,patch_Y,patch_X,model):

    h,w,c = image.shape
    Count=np.zeros((h,w))
    S=np.zeros((h,w))
    K_arr=np.zeros((h,w,c))
    T_arr=np.zeros((h,w))
    P=[]


    c=0
    for i in L1:
        for j in L2:
            patch = image[i : i+patch_Y,j:j+patch_X,:].copy()
            patch=trans.resize(patch,(b_patch_Y,b_patch_X,3))
            P.append(patch) 
            c=c+1

    P=np.array(P)        
    T, K = model.predict(P)

    c=0
    for i in L1:
        for j in L2:
           
            t=trans.resize(T[c,:,:,0],(patch_Y,patch_X))
            k=trans.resize(K[c,:,:,:],(patch_Y,patch_X,3))
            K_arr[i : i+patch_Y,j:j+patch_X,:]=K_arr[i : i+patch_Y,j:j+patch_X,:]+k
            T_arr[i : i+patch_Y,j:j+patch_X]=T_arr[i : i+patch_Y,j:j+patch_X]+t
            
            Count[i : i+patch_Y,j:j+patch_X]=Count[i : i+patch_Y,j:j+patch_X]+1

            c=c+1

                                                      
    T_arr[Count!=0] = T_arr[Count!=0]/Count[Count!=0]
    Count=Count[:,:,np.newaxis]
    Count = np.tile(Count,(1,1,3))
    K_arr[Count!=0] = K_arr[Count!=0]/Count[Count!=0]

    return T_arr,K_arr 


def new_dehaze(hazy_image_path,model):
    R_SIZE=850.0    
    image_o=read_image(hazy_image_path)
    #image_o=open_img(hazy_image_path)
    dim=min(image_o.shape[0],image_o.shape[1])

   

    if(dim>=R_SIZE):
        k=(R_SIZE/dim)
        image=trans.resize(image_o,(int(image_o.shape[0]*k),int(image_o.shape[1]*k),3))
        h,w,c = image.shape
        patch_Y=patch_X=128
        S=[patch_X*2,patch_X*3,patch_X*4]
        N=len(S)
    elif(dim<850  and dim >128):
        image=image_o
        patch_Y=patch_X=128

        S=[]
        k=1
        while(patch_X*k<min(dim,512)):
            S.append(patch_X*k)
            k=k+1
        N=len(S)
    else:
        print "ERROR:Too small to dehaze"

    s1=s2=np.ones((N,))




    h,w,c = image.shape

    Count=np.ones((N,h,w))
    STD=np.zeros((N,h,w))
    K_arr=np.zeros((N,h,w,c))
    T_arr=np.zeros((N,h,w))

    for c in range(0,N):
        patch_X=patch_Y=S[c]
        L1=range(0,h - patch_Y,patch_Y*2//4)
        L2=range(0, w - patch_X, patch_X*2//4)
        L1.append(h - patch_Y)
        L2.append(w - patch_X)  

        T,K=f(image,L1,L2,patch_Y,patch_X,model)                       

        T_arr[c]=T
        K_arr[c]=K

        c=c+1      
    



    #generate weight with sum 1
    s1=s1/np.sum(s1)
    s1=s1[:,np.newaxis,np.newaxis]    
    s2=s2/np.sum(s2)
    s2=s2[:,np.newaxis,np.newaxis]    
   
   
    #make a weight map 
    Count1=Count*s1
    Count2=Count*s2
    Count2=Count2[:,:,:,np.newaxis]

    
    #mul weight map
    K1=np.sum((K_arr*Count2),axis=0)
    T1=np.sum(T_arr*Count1,axis=0)

    #compute weight sum
    Count1_s=np.sum(Count1,axis=0)
    Count2_s=np.sum(Count2,axis=0)


    Count2_s=np.tile(Count2_s,(1,1,3))
    K1[Count2_s>0]=K1[Count2_s>0]/Count2_s[Count2_s>0]
    T1[Count1_s>0]=T1[Count1_s>0]/Count1_s[Count1_s>0]

   

    K1=smooth_A_guided_filter(image,K1,60,0.001)
    T1=guided_filter(rgb2gray(image),T1,60,0.001)

    if(R_SIZE<dim):
        T1=trans.resize(T1,(int(image_o.shape[0]),int(image_o.shape[1])))
        K1=trans.resize(K1,(int(image_o.shape[0]),int(image_o.shape[1]),3))

    return image_o,T1,K1





### reconstructig the dehazed image from K,t and original hazy image
def dehaze(t,img,K):
    if(t is False):
        return False    

    t=t[:,:,np.newaxis]
    t = np.tile(t,(1,1,3))
    J=(img-K)/np.maximum(t,0.001)

    J = J.clip(0,1)
    J[np.isnan(J)]=0

    return J
####



def dehaze_fast(hazy_image_path,model):

    img,T,K=new_dehaze(hazy_image_path,model)

    J = dehaze(T, img,K)
    return J,img,T,K

def get_trained_model(weights_path):
    model=get_model()
    model.load_weights(weights_path)
    return model 



def evaluate(model_path):
    
    #models=load_model(model_path)



    Time=0
    file_list=os.listdir(hazy_image_dir)
    if(len(file_list)==0):
        print"ERROR: There is no file in the dir ->",hazy_image_dir
        exit()


    print"---------------------------------------------------------"
    print"File name    Time(s) "
    for  d in os.listdir(hazy_image_dir):
        start = time.time()        

        t=(end-start)
        Time=Time+t
        print d+"  "+str(t)
        
        plt.imsave(out_image_dir+d,J,format="jpg")
        imsave(out_image_dir+d,J)
        imsave(out_image_dir+"A"+d,A)
        imsave(out_image_dir+"T"+d,T)

    print "average time :",Time/len(os.listdir(hazy_image_dir))






model_path=base_dir+"models/model_weights.h5"
models=get_trained_model(model_path)
J,I,T,A=dehaze_fast(hazy_image_path,models)
imsave(out_image_path+f_out,J)










