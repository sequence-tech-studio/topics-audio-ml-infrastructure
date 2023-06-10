import os
import tempfile
import traceback
import zipfile
from flask import Flask, request, send_file
from flask_restful import Api
from AudioProcessor import AudioUnmix
import json

# Create Flask app
app = Flask(__name__)
api = Api(app)

def create_output_directory(directory):
    os.makedirs(directory, exist_ok=True)


def run_audio_unmix(unmix_func, output_directory, *args):
    try:
        analysis_results = unmix_func(*args, output_directory)

        # Save analysis results to a json file
        with open(os.path.join(output_directory, 'analysis_results.json'), 'w') as f:
            json.dump(analysis_results, f)

        # Create a zip file
        output_zip_path = os.path.join('tmp', 'audio_separated.zip')
        with zipfile.ZipFile(output_zip_path, 'w') as zipf:
            for root, dirs, files in os.walk(output_directory):
                for file in files:
                    zipf.write(os.path.join(root, file))
                    os.remove(os.path.join(root, file))
        return output_zip_path

    except Exception as e:
        print(f"Exception occurred: {str(e)}")
        traceback.print_exc()
        raise

@app.route('/v1/unmix', methods=['POST'])
def unmix_audio():
    if 'audio' not in request.files:
        return "No audio file in request", 400
    file = request.files['audio']

    if file.content_length > 50 * 1024 * 1024:
        return "File size exceeds limit of 50MB", 400

    # Create a temporary file and save the uploaded file's content into it
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    file.save(temp_file.name)
    filepath = temp_file.name

    # Debugging print statement
    print(f"File saved at: {filepath}. File exists: {os.path.exists(filepath)}")

    output_directory = os.path.join('tmp', 'unmixed')  # Set the output directory path
    create_output_directory(output_directory)

    try:
        unmixer = AudioUnmix()
        output_zip_path = run_audio_unmix(unmixer.run, output_directory, filepath)

        # Removing the temporary file
        os.remove(filepath)

    except Exception as e:
        return str(e), 500

    return send_file(output_zip_path, mimetype='application/zip', as_attachment=True)

@app.route('/v1/unmix_youtube', methods=['POST'])
def unmix_youtube():
    if 'url' not in request.json:
        return "No url in request", 400
    url = request.json['url']

    output_directory = os.path.join('tmp', 'unmixed')  # Set the output directory path
    create_output_directory(output_directory)

    try:
        unmixer = AudioUnmix()
        output_zip_path = run_audio_unmix(unmixer.run_from_youtube, output_directory, url)

    except Exception as e:
        return str(e), 500

    return send_file(output_zip_path, mimetype='application/zip', as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0',port=80)