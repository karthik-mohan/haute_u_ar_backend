from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from genai import gen_vton
from werkzeug.utils import secure_filename
import os
import tempfile

#app = Flask(__name__)

app = Flask(__name__, static_folder='processed_images')

CORS(app, supports_credentials=True)
#CORS(app, supports_credentials=True, resources={r"/*": {"origins": "*"}})  # Allow requests from any originorigins=["http://localhost:3000"])

#CORS(app, resources={r"/proc": {"origins": "http://localhost:3000"}}, supports_credentials=True)
#@app.route("/proc")
@app.route('/proc', methods=['POST'])
def process_images():
    # Retrieve images from the request
    print("Request came here")
    print(request)
    print(request.headers)
    print(request.files)
   
    
    user_image_t = request.files.get('userImage')
    dress_image_t = request.files.get('dressImage')
    #print(dress_image_t.filename)
    print(user_image_t.filename)
    #file = request.files['file']
    if dress_image_t:
        # Save the file to a temporary file
        temp_dir = tempfile.gettempdir()
        filename = secure_filename(dress_image_t.filename)
        temp_path = os.path.join(temp_dir, filename)
        dress_image_t.save(temp_path)
    dress_image = temp_path
    if user_image_t:
        temp_dir = tempfile.gettempdir()
        filename = secure_filename(user_image_t.filename)
        temp_path_1 = os.path.join(temp_dir, filename)
        user_image_t.save(temp_path_1)
    user_image = temp_path_1

    gen_vton(user_image, dress_image)
    processed_image_1_path = './processed_images/output_image.jpg'
    processed_image_2_path = './processed_images/output_image_1.jpg'
    
    # Save your images using the paths above...

    # Return the URL for the saved images
    url_to_processed_image_1 = request.host_url + processed_image_1_path
    url_to_processed_image_2 = request.host_url + processed_image_2_path
    # Process images...
    # For the sake of this example, let's say the processing function returns two image URLs
    processed_image_urls = [url_to_processed_image_1, url_to_processed_image_2]
    os.remove(temp_path)
    os.remove(temp_path_1)
    return jsonify({'processedImages': processed_image_urls})

@app.route('/processed_images/<filename>')
def processed_images(filename):
    print("request_came_here")
    return send_from_directory(app.static_folder, filename)
    # Example of generating a unique filename for the output
   

#    

if __name__ == '__main__':
    app.run()
