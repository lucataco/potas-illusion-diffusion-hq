import banana_dev as client
from io import BytesIO
from PIL import Image
import gzip
import base64
import time

# Create a reference to your model on Banana
my_model = client.Client(
    api_key="",
    model_key="",
    url="http://localhost:8000",
)

# read input file
with open("spiral.png", "rb") as f:
    image_bytes = f.read()
image_encoded = base64.b64encode(image_bytes)
image = image_encoded.decode("utf-8")

# Specify the model's input JSON
inputs = {
    "prompt" : "(masterpiece:1.4), (best quality), (detailed), Medieval village scene with busy streets and castle in the distance",
    "negative_prompt": "ugly, disfigured, low quality, blurry, nsfw",
    "image": image,
    "seed": "2288773312"
}

# Call your model's inference endpoint on Banana.
t1 = time.time()
result, meta = my_model.call("/", inputs)
t2 = time.time()

result_img = result["outputs"]
image_encoded = result_img.encode('utf-8')
image_data = base64.b64decode(image_encoded)
image_data = gzip.decompress(image_data)
image_io = BytesIO(image_data)
pil_image = Image.open(image_io)
pil_image.save("output.png")
print("Time to run: ", t2 - t1)
