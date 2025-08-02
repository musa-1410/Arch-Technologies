import cv2
import numpy as np
import time

# Configuration
CONF_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4
INPUT_WIDTH = 416
INPUT_HEIGHT = 416

def load_classes():
    classes = []
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    return classes

def load_net():
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    return net

def detect_objects(frame, net, output_layers):
    blob = cv2.dnn.blobFromImage(
        frame, 
        1/255.0, 
        (INPUT_WIDTH, INPUT_HEIGHT), 
        swapRB=True, 
        crop=False
    )
    net.setInput(blob)
    outputs = net.forward(output_layers)
    return outputs

def process_outputs(frame, outputs, classes):
    height, width = frame.shape[:2]
    boxes = []
    confidences = []
    class_ids = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > CONF_THRESHOLD:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(
        boxes, confidences, 
        CONF_THRESHOLD, NMS_THRESHOLD
    )
    
    return boxes, confidences, class_ids, indices

def draw_labels(frame, boxes, confidences, class_ids, indices, classes):
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
            color = colors[class_ids[i]]
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(
                frame, label, (x, y-5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )

def main():
    classes = load_classes()
    net = load_net()
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]
    
    cap = cv2.VideoCapture(0)
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Object detection
        outputs = detect_objects(frame, net, output_layers)
        boxes, confidences, class_ids, indices = process_outputs(
            frame, outputs, classes
        )
        draw_labels(frame, boxes, confidences, class_ids, indices, classes)
        
        # Calculate FPS
        frame_count += 1
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time
        
        # Display FPS
        cv2.putText(
            frame, f"FPS: {fps:.2f}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )
        
        cv2.imshow("Real-time Object Detection", frame)
        
        # Exit on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()