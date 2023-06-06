import zipfile
import shutil
import traceback
from flask import Flask, request, send_file
from flask_restful import Api, Resource
from werkzeug.utils import secure_filename
from AudioUnmix import AudioUnmix

import os
import tempfile
# Create Flask app
app = Flask(__name__)
api = Api(app)

class UnmixResource(Resource):
    def post(self):
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

        os.makedirs(output_directory, exist_ok=True)  # Create the output directory if it doesn't exist

        # Debugging print statement
        print(f"Output directory: {output_directory}. Directory exists: {os.path.isdir(output_directory)}")

        try:
            unmixer = AudioUnmix()
            unmixer.run(filepath, output_directory)

            # Creating zip file
            output_zip_path = os.path.join('tmp', 'audio_separated.zip')
            with zipfile.ZipFile(output_zip_path, 'w') as zipf:
                for root, dirs, files in os.walk(output_directory):
                    for file in files:
                        zipf.write(os.path.join(root, file))
            
            # Removing the temporary file
            #os.remove(filepath)

        except Exception as e:
            # Log the exception message and traceback
            print(f"Exception occurred: {str(e)}")
            traceback.print_exc()
            return str(e), 500

        return send_file(output_zip_path, mimetype='application/zip', as_attachment=True)
    
class UnmixYoutubeResource(Resource):
    def post(self):
        if 'url' not in request.json:
            return "No url in request", 400
        url = request.json['url']

        output_directory = os.path.join('tmp', 'unmixed')  # Set the output directory path
        os.makedirs(output_directory, exist_ok=True)  # Create the output directory if it doesn't exist

        try:
            unmixer = AudioUnmix()
            unmixer.run_from_youtube(url, output_directory)

            # Creating zip file
            output_zip_path = os.path.join('tmp', 'audio_separated.zip')
            with zipfile.ZipFile(output_zip_path, 'w') as zipf:
                for root, dirs, files in os.walk(output_directory):
                    for file in files:
                        zipf.write(os.path.join(root, file))

        except Exception as e:
            # Log the exception message and traceback
            print(f"Exception occurred: {str(e)}")
            traceback.print_exc()
            return str(e), 500

        return send_file(output_zip_path, mimetype='application/zip', as_attachment=True)


api.add_resource(UnmixResource, '/v1/unmix')
api.add_resource(UnmixYoutubeResource, '/v1/unmix_youtube')

if __name__ == '__main__':
    app.run(debug=True)