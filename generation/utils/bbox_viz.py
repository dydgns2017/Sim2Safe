# %%
import cv2
import numpy as np
from matplotlib import pyplot as plt
# %%
def yolo_bbox_viz(image, label):
    """
    Visualizes YOLO bounding boxes on an image with class-specific colors and enhanced visibility.

    Args:
        image (str): The path to the image file.
        label (str): The path to the label file containing bounding box coordinates.

    Returns: None
    """
    # Define colors for each class (5 classes)
    class_colors = [
        (0, 255, 0),    # Class 0: Green
        (255, 255, 0),  # Class 1: Cyan
        (255, 165, 0),  # Class 2: Orange
        (255, 0, 0),    # Class 3: Red
        (128, 0, 128)   # Class 4: Purple
    ]

    image = cv2.imread(image)
    if image is None:
        raise ValueError(f"Image not found or unable to read: {image}")

    # Read the label file
    with open(label, 'r') as f:
        lines = f.readlines()

    # Loop through each line in the label file
    for line in lines:
        # Split the line into class ID and bounding box coordinates
        components = line.strip().split()
        if len(components) != 5:
            raise ValueError(f"Invalid label format: {line}")
        class_id, x_center, y_center, width, height = map(float, components)

        # Convert YOLO format to pixel coordinates
        img_height, img_width = image.shape[:2]
        x_center *= img_width
        y_center *= img_height
        width *= img_width
        height *= img_height

        # Calculate the top-left and bottom-right corners of the bounding box
        x1 = int(x_center - width / 2)
        y1 = int(y_center - height / 2)
        x2 = int(x_center + width / 2)
        y2 = int(y_center + height / 2)

        # Draw the bounding box on the image
        color = class_colors[int(class_id) % len(class_colors)]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Add a background rectangle for the class ID text
        text = f"Class {int(class_id)}"
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(image, (x1, y1 - text_height - baseline - 5), (x1 + text_width, y1), color, -1)

        # Display the class ID with larger font and better visibility
        cv2.putText(image, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
