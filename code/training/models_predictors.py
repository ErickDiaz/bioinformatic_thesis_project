import cv2
import numpy as np


def yolo_model_prediction(model_path, classes_file, model_file, weigths_file, inputs, input_width, input_height, conf_threshold = 0.4, generate_mask = True):
    '''
    model_path     :path of classes_file, model_file, weigths_file 
    classes_file   :this file contains a list of the object classes
    model_file     :cfg file
    weigths_file   :weigths file
    inputs
    '''
    
    # Load model
    net = cv2.dnn.readNetFromDarknet(model_path + model_file, model_path + weigths_file)
    
    # outputs initialization
    count  = 0
    tested = 0
    #img_index = 0
    class_ids = []
    confidences = []
    boxes = []
    images_index = []
    masks = []
    
    for img_index,i in enumerate(inputs):
        tested += 1
        image = cv2.imread(i)
        Width = image.shape[1]
        Height = image.shape[0]
        scale = 1/255

        blob = cv2.dnn.blobFromImage(image, scale, (input_width,input_height),  [0,0,0],swapRB=True, crop=False)

        net.setInput(blob)

        layer_names = net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

        outs = net.forward(output_layers)
        
        ## MASK ##
        if (generate_mask):
            masks.append(np.zeros((input_width, input_height), dtype=np.uint8))
        ##########


        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence >= conf_threshold:
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)
                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])
                    #print("confidence: {:.2f}, class: {:.2f} \n ".format(confidence, class_id))
                    images_index.append(img_index)
                    
                    ## MASK ##
                    if (generate_mask):
                        current_mask = masks[img_index]
                        current_mask[slice(round(x), round(x+w)), slice(round(y), round(y+h))] = 1
                        masks[img_index] = current_mask     
                    ##########
                    count +=1
                    

        #img_index+=1

    print("####### RESULTS ########")
    print("Total images tested: {}, Number of detections: {}".format(tested, count))

    
    return class_ids, confidences, boxes, images_index, masks