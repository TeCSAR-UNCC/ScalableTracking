import numpy as np



def load_openpose_features_train(w_img,h_img,video_frame_features,batch_size, num_steps, id):    
        
        video_frame_features = np.asarray(video_frame_features,dtype=np.float32)
        wid = w_img
        ht = h_img
        num_obj=1

        openpose_features=[]
        for frame_num in range(id,id+num_steps*batch_size):  # if id=2 gather frames from 2 to 7
            opFeatures = video_frame_features[frame_num][num_obj-1].tolist()  #frame_num,obj1,255 features
            openpose_features =openpose_features + opFeatures
        openpose_features = np.asarray(openpose_features,dtype=np.float32)
        openpose_features = np.reshape(openpose_features, [batch_size*num_steps,54])
        
        return openpose_features

def load_openpose_gt_train(w_img,h_img,video_frame_bboxes, batch_size, num_steps, id):
    video_frame_bboxes = np.asarray(video_frame_bboxes,dtype=np.float32)
    num_obj=1
    wid = w_img
    ht = h_img
    openpose_gt=[]
    
    for frame_num in range(id+num_steps,id+1+(num_steps*batch_size),num_steps):
        video_frame_bboxes[frame_num][num_obj-1][0] += (video_frame_bboxes[frame_num][num_obj-1][2]/2.0) # convert x,y to centroid 
        video_frame_bboxes[frame_num][num_obj-1][1] += (video_frame_bboxes[frame_num][num_obj-1][3]/2.0)

        video_frame_bboxes[frame_num][num_obj-1][0] = (video_frame_bboxes[frame_num][num_obj-1][0]/wid)
        video_frame_bboxes[frame_num][num_obj-1][1] = (video_frame_bboxes[frame_num][num_obj-1][1]/ht)
        video_frame_bboxes[frame_num][num_obj-1][2] = (video_frame_bboxes[frame_num][num_obj-1][2]/wid)
        video_frame_bboxes[frame_num][num_obj-1][3] = (video_frame_bboxes[frame_num][num_obj-1][3]/ht)

        #print video_frame_bboxes[frame_num][num_obj-1]
        yoloBB = video_frame_bboxes[frame_num][num_obj-1].tolist() #frame_num, obj1,4 bbox attributes
        openpose_gt = openpose_gt + yoloBB
        openpose_gt = np.asarray(openpose_gt,dtype=np.float32)
    return openpose_gt

''' EDIT THIS FOR TESTING'''
def load_openpose_features_test(w_img,h_img,filefeatures , batch_size, num_steps, id):
    video_frame_features = filefeatures
    video_frame_bboxes = np.asarray(foldb,dtype=np.float32)
    wid = w_img
    ht = h_img
    num_obj=2

    lastseen_feature=0
    lastseen_bboxes=0
    yolo_output_batched=[]
    for frame_num in range(id,id+num_steps):  # if id=2 gather frames from 2 to 7
        yolo_output_temp=[]
        for obj in range(0,num_obj):
            if  video_frame_features[frame_num][obj].all()!=0:
                lastseen_feature=frame_num
            if  video_frame_features[frame_num][obj].all()==0:
                video_frame_features[frame_num][obj]=video_frame_features[lastseen_feature][obj]
                lastseen_feature=frame_num
            yoloFeat = video_frame_features[frame_num][obj].tolist()  #frame_num,obj1,255 features
            yolo_output_temp = yolo_output_temp + yoloFeat


            if  video_frame_bboxes[frame_num][obj].all()!=0:
                lastseen_bboxes=frame_num
            if  video_frame_bboxes[frame_num][obj].all()==0:
                video_frame_bboxes[frame_num][obj]=video_frame_bboxes[lastseen_bboxes][obj]
                lastseen_bboxes=frame_num
            #video_frame_bboxes[frame_num][obj][2] += video_frame_bboxes[frame_num][obj][0] # convert w,h to bottom right 
            #video_frame_bboxes[frame_num][obj][3] += video_frame_bboxes[frame_num][obj][1]

            video_frame_bboxes[frame_num][obj][0] += (video_frame_bboxes[frame_num][obj][2]/2.0) # convert x,y to centroid 
            video_frame_bboxes[frame_num][obj][1] += (video_frame_bboxes[frame_num][obj][3]/2.0)

            video_frame_bboxes[frame_num][obj][2] -= video_frame_bboxes[frame_num][obj][0] # convert  bottom right to w,h 
            video_frame_bboxes[frame_num][obj][3] -= video_frame_bboxes[frame_num][obj][1]

            video_frame_bboxes[frame_num][obj][0] = (video_frame_bboxes[frame_num][obj][0]/wid)
            video_frame_bboxes[frame_num][obj][1] = (video_frame_bboxes[frame_num][obj][1]/ht)
            video_frame_bboxes[frame_num][obj][2] = (video_frame_bboxes[frame_num][obj][2]/wid)
            video_frame_bboxes[frame_num][obj][3] = (video_frame_bboxes[frame_num][obj][3]/ht)

            yoloBB = video_frame_bboxes[frame_num][obj].tolist() #frame_num, obj1,4 bbox attributes
            yolo_output_temp = yolo_output_temp + yoloBB
        yolo_output_batched = yolo_output_batched + yolo_output_temp

    yolo_output_batched = np.reshape(yolo_output_batched, [batch_size,num_steps,259])
    
    return yolo_output_batched



''' EDIT THIS FOR TESTING'''
def load_openpose_gt_test(w_img,h_img,arr,batch_size,num_steps,id):

    video_frame_bboxes = np.asarray(arr,dtype=np.float32)
    num_obj=2
    wid = w_img
    ht = h_img
    yolo_output_batched=[]
    lastseen_bboxes=0
    #frame_num= id+num_steps
    #if frame_num>772:
    #    return [0,0,0,0]
    for frame_num in range(id+num_steps,id+1+(num_steps),num_steps):
        yolo_output_temp=[]
        for obj in range(0,num_obj):
            # video_frame_bboxes[frame_num][obj][2] += video_frame_bboxes[frame_num][obj][0] # convert w,h to bottom right 
            # video_frame_bboxes[frame_num][obj][3] += video_frame_bboxes[frame_num][obj][1]

            if  video_frame_bboxes[frame_num][obj].all()!=0:
                lastseen_bboxes=frame_num
            if  video_frame_bboxes[frame_num][obj].all()==0:
                video_frame_bboxes[frame_num][obj]=video_frame_bboxes[lastseen_bboxes][obj]
                lastseen_bboxes=frame_num

            video_frame_bboxes[frame_num][obj][0] += (video_frame_bboxes[frame_num][obj][2]/2.0) # convert x,y to centroid 
            video_frame_bboxes[frame_num][obj][1] += (video_frame_bboxes[frame_num][obj][3]/2.0)

            video_frame_bboxes[frame_num][obj][2] -= video_frame_bboxes[frame_num][obj][0] # convert bottom right to w,h 
            video_frame_bboxes[frame_num][obj][3] -= video_frame_bboxes[frame_num][obj][1]

            video_frame_bboxes[frame_num][obj][0] = (video_frame_bboxes[frame_num][obj][0]/wid)
            video_frame_bboxes[frame_num][obj][1] = (video_frame_bboxes[frame_num][obj][1]/ht)
            video_frame_bboxes[frame_num][obj][2] = (video_frame_bboxes[frame_num][obj][2]/wid)
            video_frame_bboxes[frame_num][obj][3] = (video_frame_bboxes[frame_num][obj][3]/ht)


            yoloBB = video_frame_bboxes[frame_num][obj].tolist() #frame_num, obj1,4 bbox attributes
            yolo_output_temp = yolo_output_temp + yoloBB
        yolo_output_batched = yolo_output_batched + yolo_output_temp
    yolo_output_batched = np.reshape(yolo_output_batched, [batch_size,4])
    return yolo_output_batched