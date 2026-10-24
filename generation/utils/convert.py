import numpy as np

def get_bbox_from_txt(box_str, image_size):
    """
    Parse string to get bounding box
    """
    h, w = image_size
    box = [float(b) for b in box_str]
    x1 = (box[0] - box[2]/2) * w
    y1 = (box[1] - box[3]/2) * h
    x2 = (box[0] + box[2]/2) * w
    y2 = (box[1] + box[3]/2) * h
    return [x1, y1, x2, y2]
    
def convert_label_to_cleanlab_format(text_path, image_size):
    text_lines = list()
    with open(text_path, 'r') as f:
        text_lines = f.readlines()
    labels = list()
    bboxes = list()
    for line in text_lines:
        line_parts = line.split(" ")
        label = int(line_parts[0])
        bbox = get_bbox_from_txt(line_parts[1:], image_size)
        labels.append(label)
        bboxes.append(bbox)
    return {"labels": np.array(labels),
           "bboxes": np.array(bboxes)}

def convert_preds_to_cleanlab_format(result):
    """
    Receives a ultralytics.engine.results.Results object.
    Return prediction probabilities in cleanlab format.    
    """
    result = result.cpu()
    class_ids = list(result.names.keys())
    predictions = [np.empty(shape=(0, 5), dtype=np.float32)] * len(class_ids)

    boxes = result.boxes
    predicted_classes = boxes.cls.int()
    for class_id in class_ids:
        class_id_indices = np.where(predicted_classes == class_id)[0]
        if class_id_indices.size:
            confs = boxes.conf[class_id_indices]
            bboxes = boxes.xyxy[class_id_indices] # x1, y1, x2, y2
            class_preds = np.c_[bboxes, confs]
            predictions[class_id] = class_preds

    predictions = np.array(predictions, dtype=np.object_) # If numpy array is required
    return predictions
