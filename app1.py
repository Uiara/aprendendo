from os.path import isfile, join
from datetime import datetime
#from sklearn.preprocessing import Normalizer
#from sklearn.svm import SVC
from datetime import datetime
from imutils import paths
#from sklearn.preprocessing import LabelEncoder
#from sklearn.model_selection import GridSearchCV
from os import listdir
from os.path import isdir
from PIL import Image
from numpy import savez_compressed
from numpy import asarray
from numpy import load
from numpy import expand_dims
#from keras.models import load_model

import face_recognition
import time
import requests
import cv2
import os
import numpy as np
from uuid import getnode as get_mac

def clean_folder(path):
    filelist = [ f for f in os.listdir(path) if f.endswith(".jpg") ]
    for f in filelist:
        os.remove(os.path.join(path, f))


#id_names -> name list
#face_image -> face list
#time_OUT
#path_out -> path to dataset_out folder
#person_bed_out_details -> check_OUT file
#url
#send -> 'True' sends and 'False' doesn't send to respective url

def hig(image_hig, path_hig, url, send = False):
    time_aux = time.time()
    
    mac = get_mac()
    macString = 'XX:'+':'.join(("%012X" % mac)[i:i+2] for i in range(2, 12, 2))
    print(macString)
    
    
    FILENAME = path_hig + str(time_aux) +'_'+macString+".jpg"
    cv2.imwrite(FILENAME, image_hig)
    
    #send to higienize
    if send == True:
        files = {"file":(str(time_aux) +'_'+macString+'_'+".jpg",
                         open(path_hig + str(time_aux) +'_'+macString+".jpg","rb"),'application-type')}
        response = requests.post(url, files=files)
        if response.status_code == 200:
            print("status: ",response.status_code," - foto enviada"," ### Higienização ###")
        else:
            print("status: ",response.status_code,"Foto não enviada"," ### Higienização ###")





def check_OUT(person_identified_details,face_image,time_OUT,path_out,person_bed_out_details, path_send):

    mac = get_mac()
    macString = ':'.join(("%012X" % mac)[i:i+2] for i in range(0, 12, 2))

    #loop out the face detection to verify time and get time check-OUT
    for nb,ids in enumerate(person_identified_details):

        #after time_OUT seconds we will save the last face_image and send it within timestamp
        if time.time() - person_identified_details[nb][1] > time_OUT:
                        
            #read files there
            fn = [face_names[0] for face_names in face_image]

            #get timestamp 
            time_aux = time.time() - time_OUT

            #add in person_bed_out_details list and write the image in leito folder
            person_bed_out_details.append(person_identified_details[nb][0]+ \
            ','+'OUT'+','+str(time_aux)+','+'LEITO'+"\n")               
            cv2.imwrite(os.path.join(path_out,person_identified_details[nb][0]+','+'OUT'+','+str(time_aux)+','+ \
            'LEITO'+','+macString+".jpg"), face_image[fn.index(person_identified_details[nb][0])][1])
            cv2.imwrite(os.path.join(path_send,person_identified_details[nb][0]+'_'+'OUT'+'_'+str(time_aux)+'_'+ \
            'LEITO'+'_'+macString+".jpg"), face_image[fn.index(person_identified_details[nb][0])][1])

                        
            #remove after check-OUT
            face_image.remove(face_image[fn.index(person_identified_details[nb][0])])
            person_identified_details.remove(person_identified_details[nb]) 

def rgb_small(frame,fx,fy):

    small_frame = cv2.resize(frame, (0, 0), fx=fx, fy=fy)
    return (small_frame[:, :, ::-1])
    
def face_encoding_match(face_encodings,known_face_encodings,known_face_names,pp):

    try:
        face_detected = []

        for face_encoding in face_encodings:
            #
            #known_face_encodings = np.asarray(known_face_encodings)
            #See if the face is a match for the known face(s)
            name = "unknown"

            try:
                in_encoder = Normalizer(norm='l2')
                trainX = in_encoder.transform(asarray(known_face_encodings))
                newTrainX = in_encoder.transform(face_encoding.reshape(1, -1))
                # label encode targets
                out_encoder = LabelEncoder()
                out_encoder.fit(known_face_names)
                trainy = out_encoder.transform(known_face_names)

                ###
                #matches = face_recognition.compare_faces(trainX, newTrainX,tolerance=0.9)
                #use the known face with the smallest distance to the new face
                #face_distances = face_recognition.face_distance(trainX, newTrainX)
                #print(face_distances)

                #best_match_index = np.argmin(face_distances)
                #if matches[best_match_index]:
                #   name = known_face_names[best_match_index]
                #   print(name)

                ###
                
                # fit model
                model = SVC(kernel='linear')
                model.fit(trainX, trainy)
                # predict
                #yhat_train = model.predict(trainX)
                yhat_test = model.predict(newTrainX)
                predict_names = out_encoder.inverse_transform(yhat_test)
                ###
                name = predict_names[0]
                print("---",name,"---")
                
                yhat_prob = model.predict_proba(newTrainX)
                # get name
                class_index = yhat_test[0]
                class_probability = yhat_prob[0,class_index] * 100
                name = name+'_'+class_probability
                #face_id_names.append(name)
                #if float(class_probability) < 25:
                #   name = "unknown"
                #else:
                #   print("--------------",name,class_probability,"--------------")
                #   pp.append(name+" - "+str(class_probability)+"\n")"""


            except:
                
                print("Nenhuma pessoa cadastrada")

            face_detected.append(name)
            

        return (face_detected)

    except:
        print('Nenhuma face identificada')
        return([])

def saving_faces_detected(person_identified_details,face_image,frame,name,face_par,top,bottom,left,right, \
    frame_aux,known_face_encodings,known_face_names):
    #get current faces detected 
    for nb,ids in enumerate(person_identified_details): 
        if person_identified_details[nb][0] == name:    
            person_identified_details[nb][1] = time.time()
                
            aux_top = max(0, top - int((face_par+0.1)*(bottom-top)))
            aux_bottom = min(bottom + int((face_par)*(bottom-top)),frame.shape[0])
            aux_left = max(0, left - int((face_par)*(bottom-top)))
            aux_right = min(right + int((face_par)*(bottom-top)),frame.shape[1])
                
            fn = [name_face[0] for name_face in face_image]
            face_image[fn.index(person_identified_details[nb][0])][1] = frame_aux[aux_top:aux_bottom,aux_left:aux_right]
        
            """try: 
                if person_identified_details[nb][2] < 0:
                    cv2.imwrite(str(person_identified_details[nb][2])+'.jpg',frame_aux[aux_top:aux_bottom,aux_left:aux_right])      
                    dict_ = {}
                    aux2 = time.time()
        
                    #dictionary to make recognize processing dynamic
                    dict_[aux2] = face_recognition.load_image_file(str(person_identified_details[nb][2])+".jpg") 
        
                    aux3 = time.time()
        
                    #dict_[aux3] = face_recognition.face_encodings(dict_[aux2])[0]
                    #print(dict_[aux3],"############_encodings")
                    
                    #dict_[aux3] = face_embeddings(dict_[aux2])[0]
                    #print(dict_[aux3],"############_embeddings")
                    
                    #known_face_encodings.append(dict_[aux3])

                    face_encoding_aux = face_embeddings(frame = frame_aux,model = model,face_locations = [(top, right, bottom, left)])
        
                    known_face_encodings.append(face_encoding_aux[0])

                    known_face_names.append(person_identified_details[nb][0])
                    person_identified_details[nb][2] += 1
                    print("Nova imagem adicionada ao "+person_identified_details[nb][0]+" - "+ \
                    str(person_identified_details[nb][2]))   

            except:
                os.remove("__.jpg")
                print("Imagem não adicionada ao "+person_identified_details[nb][0])"""
                

def send_request_image(frame_aux, path_send,required_size=(160, 160)):
    mac = get_mac()
    macString = ':'.join(("%012X" % mac)[i:i+2] for i in range(0, 12, 2))
    
    time_aux = time.time()
    home = "/home/pi/higienize_rpi/"
    
    cv2.imwrite(os.path.join(home , '0_' + str(time_aux)+'_'+'LEITO'+'_'+macString+".jpg"),frame_aux)
    img = cv2.imread(os.path.join(home,'0_' + str(time_aux)+'_'+'LEITO'+'_'+macString+".jpg"))
    
    resized = cv2.resize(img, required_size, interpolation = cv2.INTER_AREA)
    cv2.imwrite(os.path.join(path_send,'0_' +str(time_aux)+'_'+'LEITO'+'_'+macString+".jpg"),resized)
    os.remove(os.path.join(home, '0_' + str(time_aux)+'_'+'LEITO'+'_'+macString+".jpg"))
    
def save_detected_face(face_par,top,bottom,left,right,frame_aux, faces, path_send, required_size=(160, 160)):
    
    mac = get_mac()
    macString = ':'.join(("%012X" % mac)[i:i+2] for i in range(0, 12, 2))

	#the same process to save new check-OUT from person already identified
    time_aux = time.time()

    aux_top = max(0, top - int((face_par+0.1)*(bottom-top)))
    aux_bottom = min(bottom + int((face_par)*(bottom-top)),frame_aux.shape[0])
    aux_left = max(0, left - int((face_par)*(bottom-top)))
    aux_right = min(right + int((face_par)*(bottom-top)),frame_aux.shape[1])
    
    new_in_img = frame_aux[aux_top:aux_bottom,aux_left:aux_right]
    #new_in_img = new_in_img.resize(required_size)
    home = "/home/pi/higienize_rpi/"
    cv2.imwrite(os.path.join(home , str(faces) + '_' + str(time_aux)+'_'+'LEITO'+'_'+macString+".jpg"),new_in_img)
    img = cv2.imread(os.path.join(home,str(faces) + '_' + str(time_aux)+'_'+'LEITO'+'_'+macString+".jpg"))
    
    resized = cv2.resize(img, required_size, interpolation = cv2.INTER_AREA)
    cv2.imwrite(os.path.join(path_send,str(faces) + '_' +str(time_aux)+'_'+'LEITO'+'_'+macString+".jpg"),resized)
    os.remove(os.path.join(home,str(faces) + '_' + str(time_aux)+'_'+'LEITO'+'_'+macString+".jpg"))



def new_check_IN(name,person_identified_details,face_image,person_bed_in_details,path_in,face_par,
    top,bottom,left,right,frame_aux, path_send):

    mac = get_mac()
    macString = ':'.join(("%012X" % mac)[i:i+2] for i in range(0, 12, 2))
    
    if name not in [names[0] for names in person_identified_details]:
        person_identified_details.append([name,time.time(),0])                      
        aux_top = max(0, top - int((face_par+0.1)*(bottom-top)))
        aux_bottom = min(bottom + int((face_par)*(bottom-top)),frame_aux.shape[0])
        aux_left = max(0, left - int((face_par)*(bottom-top)))
        aux_right = min(right + int((face_par)*(bottom-top)),frame_aux.shape[1])
                            
        #grab face image
        face_image.append([name,frame_aux[aux_top:aux_bottom,aux_left:aux_right]])
    
        #onlyfiles = [f for f in listdir(path_in) if isfile(join(path_in, f))]
        #the same process to save new check-OUT from person already identified
        time_aux = time.time()
        
        new_in_img = frame_aux[aux_top:aux_bottom,aux_left:aux_right]   
        cv2.imwrite(os.path.join(path_in,name+','+'IN'+','+str(time_aux)+','+'LEITO'+','+macString+".jpg"),new_in_img)
        person_bed_in_details.append(name+','+'IN'+','+str(time_aux)+','+'LEITO'+"\n")
        
        cv2.imwrite(os.path.join(path_send,name+'_'+'IN'+'_'+str(time_aux)+'_'+'LEITO'+'_'+macString+".jpg"),new_in_img)
        


def first_check_IN(person_identified_details,face_image,person_bed_in_details,path_in,face_par, \
    top,bottom,left,right,frame_aux,time_check,cnt, \
    known_face_encodings,known_face_names,url,model,send = False):

    mac = get_mac()
    macString = ':'.join(("%012X" % mac)[i:i+2] for i in range(0, 12, 2))
        
    dict_ = {}      
    unk = True
    aux_top = max(0, top - int((face_par+0.1)*(bottom-top)))
    aux_bottom = min(bottom + int((face_par)*(bottom-top)),frame_aux.shape[0])
    aux_left = max(0, left - int((face_par)*(bottom-top)))
    aux_right = min(right + int((face_par)*(bottom-top)),frame_aux.shape[1])
                        
    #grab face image
    image_unknown = frame_aux[aux_top:aux_bottom,aux_left:aux_right]

    time_aux = time.time()                  
        
    #list to control check-IN and check-OUT
    person_identified_details.append(["ID"+"-"+str(cnt),time_aux,0])
    #list to grab check-IN details
    person_bed_in_details.append("ID"+"-"+str(cnt)+','+'IN'+','+str(time_aux)+','+'LEITO'+"\n")
    face_image.append(["ID"+"-"+str(cnt),frame_aux[aux_top:aux_bottom,aux_left:aux_right]])
    #saving face image
    cv2.imwrite(os.path.join(path_in,"ID"+"-"+str(cnt)+','+'IN'+','+str(time_aux)+','+'LEITO'+','+macString+".jpg"), image_unknown)
    
    try:            
        aux2 = "ID" + "-" + str(cnt) + "-_-"
        
        #dictionary to make recognize processing dynamic
        dict_[aux2] = face_recognition.load_image_file(path_in+"ID"+"-"+str(cnt)+','+'IN'+','+str(time_aux)+','+'LEITO'+','+macString+".jpg") 

        aux3 = "ID" + "-" + str(cnt) + "^_^"
        
        #dict_[aux3] = face_recognition.face_encodings(dict_[aux2])[0]
        
        face_encoding_aux = face_embeddings(frame = frame_aux,model = model,face_locations = [(top, right, bottom, left)])
        
        known_face_encodings.append(face_encoding_aux[0])
        known_face_names.append("ID" + "-" + str(cnt))

        if send == True:
            files = {"file":("ID"+"-"+str(cnt)+'_'+'IN'+'_'+str(time_aux) \
            +'_'+'LEITO'+'_'+macString+'_'+".jpg",open(path_in+"ID"+"-"+str(cnt)+','+'IN'+','+str(time_aux) \
            +','+'LEITO'+','+macString+".jpg","rb"),'application-type')}                        
            response = requests.post(url, files=files)
            print(response.status_code)
        return(True)


    except:
        print("face detectada, mas não cadastrada - nº: "+ str(cnt))            
        os.remove(path_in+"ID"+"-"+str(cnt)+','+'IN'+','+str(time_aux)+','+'LEITO'+".jpg")
        person_identified_details.remove(["ID"+"-"+str(cnt),time_aux])
        person_bed_in_details.remove("ID"+"-"+str(cnt)+','+'IN'+','+str(time_aux)+','+'LEITO'+"\n")
        face_image.remove(["ID"+"-"+str(cnt),frame_aux[aux_top:aux_bottom,aux_left:aux_right]])
        known_face_encodings.remove(face_encoding_aux[0])
        known_face_names.remove("ID" + "-" + str(cnt))
        return(False)

def get_embedding(model, face_pixels):
    # scale pixel values
    face_pixels = face_pixels.astype('float32')
    # standardize pixel values across channels (global)
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    # transform face into one sample
    samples = expand_dims(face_pixels, axis=0)
    # make prediction to get embedding
    yhat = model.predict(samples)

    return yhat[0]

def face_embeddings(frame,model,face_locations):

    #cv2.imwrite('aux.jpg',frame_aux)
    image = frame
    image = image[:, :, ::-1]
        
    #image = Image.open(filename)
    
    # convert to RGB, if needed
    #image = image.convert('RGB')

    # convert to array
    pixels = asarray(image)

    # convert each face in the train set to an embedding
    newTrainX = list()

    for _,(top, right, bottom, left) in enumerate(face_locations):

        # y1 = max(0, face_locations[0][0])
        # x2 = min(face_locations[0][1],pixels.shape[1])
        # y2 = min(face_locations[0][2],pixels.shape[0])
        # x1 = max(0, face_locations[0][3])

        y1 = max(0, top)
        x2 = min(right,pixels.shape[1])
        y2 = min(left,pixels.shape[0])
        x1 = max(0, bottom)

        # extract the face
        face = pixels[y1:y2, x1:x2]

        # resize pixels to the model size
        image = Image.fromarray(face)

        image = image.resize((160, 160))

        face_array = asarray(image)

        X = list()

        # store
        X.extend(face_array)

        X = asarray(X)

        trainX = X

        # load the facenet model
        #model = load_model(path_keras_model)
        model = model
        
        embedding = get_embedding(model, trainX)

        newTrainX.append(embedding)
        
    newTrainX = np.asarray(newTrainX)
    #print(newTrainX)
    return(newTrainX)

def create_unknown_id(path,model,known_face_encodings,known_face_names):

    for i in range(1):
        #Grab a single frame of video
        image = cv2.imread(path+str(i)+".jpg")
        
        #Resize frame of video for faster face recognition processing
        #Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        
        face_locations = face_recognition.face_locations(image, model = "hog")
        
        face_encoding_aux = face_embeddings(frame = image,model = model,face_locations = face_locations)
            
        known_face_encodings.append(face_encoding_aux[0])
        known_face_names.append("unknown")

def get_name_from_dataset(url):
    """get names from dataset in higienize"""

    import requests
    import re

    #get
    response = requests.get(url)

    #print("response status: ",response.status_code)
    #print("response text: ",response.text)
    #print("response headers: ",response.headers)
    start = '\"'
    end = '\"'
    Names = []
    for i in response.text.split(","):
        s = i
        Names.append(s[s.find(start)+len(start):s.rfind(end)])
        
    return Names
    
def get_array_from_dataset(url):
    """get arrays from dataset in higienize"""

    import requests
    import re
    import numpy
    
    #get
    response = requests.get(url)

    Encodings = []
    encodings = []

    for j in range(len(response.text.split("array"))-1):
        encodings = []
        for i in str(response.text.split("array")[j+1]).split(","):
            aux=re.sub('[n( []', '',i)
            #print(aux.replace('\\',''))
            try:
                encodings.append(float(aux.replace('\\','')))
            except:
                try:
                    encodings.append(float(aux.replace(']','')))
                except:
                    pass
                #print("No float")
        encodings = numpy.array(encodings,dtype='float32')
        Encodings.append(encodings)

    return Encodings



# extract a single face from a given photograph
def extract_face(i,filename,locations,required_size=(160, 160)):
    # load image from file
    image = Image.open(filename)
    # convert to RGB, if needed
    image = image.convert('RGB')
    # convert to array
    pixels = asarray(image)

    #face_locations = face_recognition.face_locations(pixels, model = "hog")
    ###
    face_locations = locations

    top = int(face_locations[i][0])
    right = int(face_locations[i][1])
    bottom = int(face_locations[i][2])
    left = int(face_locations[i][3])

    face_par = 0

    y1 = max(0, top - face_par)
    y2 = min(bottom + face_par,pixels.shape[0])
    x1 = max(0, left - face_par)
    x2 = min(right + face_par,pixels.shape[1])
    #print(y1,y2,x1,x2,pixels.shape)
    
    #y1 = max(0, top)
    #x2 = min(right,pixels.shape[1])
    #y2 = min(left,pixels.shape[0])
    #x1 = max(0, bottom)

    # extract the face
    #face = pixels[y1:x1,y2:x2]
    #
    face = pixels[y1:y2,x1:x2]

    # resize pixels to the model size
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = asarray(image)
    #cv2.imwrite('dale.jpg',face_array)
    return face_array

# load images and extract faces for all images in a directory
def load_faces_rec(i,directory,locations):
    faces = list()
    # enumerate files
    try:
        face = extract_face(i,directory,locations)
        # store
        faces.append(face)
    except:
        print("face não identificada")

    return faces

# load a dataset that contains one subdir for each class that in turn contains images
def load_image_rec(i,directory,locations):
    X = list()
    faces = load_faces_rec(i,directory,locations)

    # store
    X.extend(faces)

    return asarray(X)

def face_rec(i,model,image,known_face_names,known_face_encodings,locations):
            
    trainX = load_image_rec(i,image,locations)
    if len(trainX) == 0:
        return("FaceNotDetected")
    
    # load the facenet model
    #model = load_model(path_keras_model)
    
    # convert each face in the train set to an embedding
    newTrainX = list()
    #print(trainX[0])
    embedding = get_embedding(model, trainX[0])
    newTrainX.append(embedding)
    newTrainX = asarray(newTrainX)

    trainX, trainy = asarray(known_face_encodings),asarray(known_face_names)

    # normalize input vectors
    in_encoder = Normalizer(norm='l2')
    trainX = in_encoder.transform(trainX)
    newTrainX = in_encoder.transform(newTrainX)
    # label encode targets
    out_encoder = LabelEncoder()
    out_encoder.fit(trainy)
    trainy = out_encoder.transform(trainy)

    # fit model
    model = SVC(kernel='linear', probability=True)
    model.fit(trainX, trainy)
    # predict
    #yhat_train = model.predict(trainX)
    yhat_test = model.predict(newTrainX)
    predict_names = out_encoder.inverse_transform(yhat_test)

    yhat_prob = model.predict_proba(newTrainX)
    # get name
    class_index = yhat_test[0]
    class_probability = yhat_prob[0,class_index] * 100

    if class_probability > 10:
        return (predict_names[0])
    else:   
        return ("unknown")

    #return(predict_names[0]+'_'+str(class_probability))
    
def opencv_dnn_face_location(filename, path_dnn_face_detector, miniSize):
    # load image from file
    image = cv2.imread(filename)
    #face detection confidence. All face detections under this value are discarded
    confidence_ = 0.65
    
    #list to (x, y)-coordinates of the bounding box for the face
    boxes = []
    face_locations = []
    #path to files needed to face detector
    protoPath = os.path.sep.join([path_dnn_face_detector, "deploy.prototxt"])
    modelPath = os.path.sep.join([path_dnn_face_detector, "res10_300x300_ssd_iter_140000.caffemodel"])

    #load files to face detector
    detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
    
    #grab the image dimensions
    (h, w) = image.shape[:2]

    #construct a blob from the image
    imageBlob = cv2.dnn.blobFromImage(
    cv2.resize(image, (300, 300)), 1.0, (300, 300),
    (104.0, 177.0, 123.0))

    #apply OpenCV's deep learning-based face detector to localize
    #faces in the input image
    detector.setInput(imageBlob)
    detections = detector.forward()
    print(detections.shape)
    #loop over the detections
    for i in range(0, detections.shape[2]):
        #extract the confidence (i.e., probability) associated with the
        #prediction
        confidence = detections[0, 0, i, 2]
#        print(confidence, "confidence")
        #filter out weak detections
        if confidence > confidence_:
            #compute the (x, y)-coordinates of the bounding box for the face
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            if (startY <= 1.1*h and endY <= 1.1*h and endX <= 1.1*w and startX <= 1.1*w):
                #reordering coordenates to use in face recognition
                boxes.append((startY, endX, endY, startX))
                for (startY, endX, endY, startX) in boxes:
                    if (endX-startX) > miniSize and (endY-startY) > miniSize:
                        face_locations.append((startY, endX, endY, startX))

    return face_locations
    
