import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from paddleocr import PaddleOCR


def load_model(use_gpu=False):
    """
    Initialize and return a PaddleOCR model
    Args:
        use_gpu (bool): Whether to use GPU
    Returns:
        PaddleOCR model instance
    """
    return PaddleOCR(
        use_angle_cls=True,
        lang="en",
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
    """
    start_time = time.time()
    ocr_results = model.ocr(image_path)
    inference_time = time.time() - start_time
    print(f"\nInference Time: {inference_time:.4f} seconds")

    detected_regions = []
    for text_regions in ocr_results:
        for region in text_regions:
            bbox_points, (text, confidence) = region
            points_array = np.array(bbox_points, dtype=np.float32)
            x, y, w, h = map(int, cv2.boundingRect(points_array))

            detected_regions.append(
                {
                    "text": text,
                    "confidence": confidence,
                    "bbox": {"x": x, "y": y, "width": w, "height": h},
                }
            )

    return detected_regions


def visualize_results(img: str, detected_text: list):
    img = plt.imread(img)

    plt.figure(figsize=(15, 15))
    plt.imshow(img)

    for item in detected_text:
        bbox = item["bbox"]
        x, y = bbox["x"], bbox["y"]
        w, h = bbox["width"], bbox["height"]

        plt.plot([x, x + w, x + w, x, x], [y, y, y + h, y + h, y], "r-", linewidth=2)

        plt.text(
            x,
            y - 5,
            f"{item['text']} ({item['confidence']:.2f})",
            bbox=dict(facecolor="white", alpha=0.7),
            fontsize=8,
        )

    plt.axis("off")
    plt.show()


# Example usage
if __name__ == "__main__":
    model = load_model()

    image_path = "sample_image.png"
    detected_text = inference(model, image_path)

    print(detected_text)

    visualize_results(image_path, detected_text)

    # # Print detected text and bounding boxes
    # print("\nDetected Text and Bounding Boxes:")
    # for item in detected_text:
    #     print(f"Text: {item['text']}")
    #     print(f"Confidence: {item['confidence']:.2f}")
    #     print(f"Bounding Box: {item['bbox']}")
    #     print("---")
