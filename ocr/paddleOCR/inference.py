# First install the required package
# pip install "paddleocr>=2.7.0"

import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from paddleocr import PaddleOCR


def load_model(use_gpu=False, lang="en"):
    """
    Initialize and return a PaddleOCR model
    Args:
        use_gpu (bool): Whether to use GPU
        lang (str): Language code for the model
    Returns:
        PaddleOCR model instance
    """
    return PaddleOCR(
        use_angle_cls=True,
        lang=lang,
        use_gpu=use_gpu,
        ocr_version="PP-OCRv4",
        show_log=False,
    )


def inference(model, image_path):
    """
    Perform OCR on an image and return detected text with positions
    Args:
        model: PaddleOCR model instance
        image_path (str): Path to the image file
    Returns:
        list: List of dictionaries containing:
            - text: The detected text string
            - confidence: Detection confidence score
            - bbox: Dictionary with x, y, width, height of the bounding box
        float: Inference time in seconds
    """
    start_time = time.time()
    result = model.ocr(image_path)
    inference_time = time.time() - start_time

    detected_text = []
    for idx in range(len(result)):
        res = result[idx]
        for line in res:
            position = line[0]
            text = line[1][0]
            confidence = line[1][1]

            # Calculate bounding box
            points = np.array(position, dtype=np.float32)
            rect = cv2.boundingRect(points)
            bbox = {
                "x": int(rect[0]),
                "y": int(rect[1]),
                "width": int(rect[2]),
                "height": int(rect[3]),
            }

            detected_text.append(
                {
                    "text": text,
                    "confidence": confidence,
                    "bbox": bbox,
                }
            )

    return detected_text, inference_time


def get_text_in_region(detected_text, x, y, width, height):
    """
    Filter text detections that fall within a specified region
    """
    results = []
    for item in detected_text:
        bbox = item["bbox"]
        # Check if the center of the text box falls within the region
        text_center_x = bbox["x"] + bbox["width"] / 2
        text_center_y = bbox["y"] + bbox["height"] / 2

        if x <= text_center_x <= x + width and y <= text_center_y <= y + height:
            results.append(item)
    return results


def visualize_results(img, detected_text, show_upright_bbox=True):
    """
    Visualize detection results on the image
    """
    plt.figure(figsize=(15, 15))
    plt.imshow(img)

    for item in detected_text:
        # Draw original quadrilateral bounding box (red)
        position = item["position"]
        plt.plot(
            [position[0][0], position[1][0]], [position[0][1], position[1][1]], "r-"
        )
        plt.plot(
            [position[1][0], position[2][0]], [position[1][1], position[2][1]], "r-"
        )
        plt.plot(
            [position[2][0], position[3][0]], [position[2][1], position[3][1]], "r-"
        )
        plt.plot(
            [position[3][0], position[0][0]], [position[3][1], position[0][1]], "r-"
        )

        if show_upright_bbox:
            # Draw minimal upright bounding rectangle (blue)
            bbox = item["bbox"]
            x, y = bbox["x"], bbox["y"]
            w, h = bbox["width"], bbox["height"]
            plt.plot([x, x + w, x + w, x, x], [y, y, y + h, y + h, y], "b--", alpha=0.5)

        # Add text annotation
        plt.text(
            position[0][0],
            position[0][1],
            f"{item['text']} ({item['confidence']:.2f})",
            bbox=dict(facecolor="white", alpha=0.7),
        )

    plt.axis("off")
    plt.show()


# Example usage
if __name__ == "__main__":
    # Initialize OCR
    model = load_model()

    # Process an image
    image_path = "sample_image.png"
    detected_text, inference_time = inference(model, image_path)

    # Print inference time
    print(f"\nInference Time: {inference_time:.4f} seconds")

    # Print detected text and bounding boxes
    print("\nDetected Text and Bounding Boxes:")
    for item in detected_text:
        print(f"Text: {item['text']}")
        print(f"Confidence: {item['confidence']:.2f}")
        print(f"Bounding Box: {item['bbox']}")
        print("---")
