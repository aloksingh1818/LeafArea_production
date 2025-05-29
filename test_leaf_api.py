import requests

# Path to a sample image (update this to a real image with a calibration object)
IMAGE_PATH = 'sample_leaf.jpg'

url = 'http://localhost:5000/analyze_leaf'

with open(IMAGE_PATH, 'rb') as img_file:
    files = {'image': img_file}
    response = requests.post(url, files=files)

if response.ok:
    print('API Response:')
    print(response.json())
else:
    print('Error:', response.status_code)
    print(response.text)
