import time

from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="OCR", device="gpu:0")
start_time = time.perf_counter()
output = pipeline.predict(
    [
        "https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_002.png"
    ]
)
end_time = time.perf_counter()
print(f"Time taken: {(end_time - start_time) * 1000:.6f} ms")
# for res in output:
#     res.print()
#     res.save_to_img("./output/")
#     res.save_to_json("./output/")
