import os
from flask import Flask, request, jsonify, send_file, render_template
from werkzeug.utils import secure_filename
from colorizers import *
import matplotlib.pyplot as plt
app = Flask(__name__)

# Set allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/colorize', methods=['POST'])
def colorize():
    if 'image' not in request.files:
        return jsonify({'error': 'No image found in request'}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join('uploads', filename)
        file.save(filepath)
        
        # Load colorizers
        colorizer_eccv16 = eccv16(pretrained=True).eval()
        colorizer_siggraph17 = siggraph17(pretrained=True).eval()
        
        # Preprocess image
        img = load_img(filepath)
        (tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256, 256))
        
        # Colorize image using ECCV16 and SIGGRAPH17 models
        out_img_eccv16 = postprocess_tens(tens_l_orig, colorizer_eccv16(tens_l_rs).cpu())
        out_img_siggraph17 = postprocess_tens(tens_l_orig, colorizer_siggraph17(tens_l_rs).cpu())
        
        # Save colorized images
        filename_eccv16 = f'eccv16_{filename}'
        filepath_eccv16 = os.path.join('uploads', filename_eccv16)
        plt.imsave(filepath_eccv16, out_img_eccv16)
        
        filename_siggraph17 = f'siggraph17_{filename}'
        filepath_siggraph17 = os.path.join('uploads', filename_siggraph17)
        plt.imsave(filepath_siggraph17, out_img_siggraph17)
        
        # Pass file paths to the template
        return render_template('index.html', 
                               colored_image_eccv16=filepath_eccv16, 
                               colored_image_siggraph17=filepath_siggraph17)

    return jsonify({'error': 'Invalid file format'}), 400

if __name__ == '__main__':
    app.run(debug=True)
