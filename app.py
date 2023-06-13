import os
import tempfile
import traceback
from flask import Flask, request
from AudioProcessor import AudioUnmix
import uuid
from datetime import datetime
from firebase_admin import credentials, firestore
from google.cloud import storage
from firebase_admin import initialize_app
import json
from flask_restful import Api

service_account_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
if service_account_path is None:
    raise ValueError("GOOGLE_APPLICATION_CREDENTIALS environment variable is not set")

cred = credentials.Certificate(service_account_path)
initialize_app(cred)

db = firestore.client()
storage_client = storage.Client()

app = Flask(__name__)
api = Api(app)


def create_output_directory(directory):
    os.makedirs(directory, exist_ok=True)


def upload_to_firebase_storage(source_file_name, destination_blob_name):
    bucket = storage_client.bucket("audio-unmix")
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)
    return blob.public_url


def write_to_firestore(document_id, chatId, data):
    chats_ref = db.collection('chats')
    chat_ref = chats_ref.document(chatId)
    messages_ref = chat_ref.collection('messages')

    message_ref = messages_ref.document(document_id)
    message_ref.set(data)


def audio_to_message(analysis_results, output_directory, chatId):
    try:
       
        messages = []
        for root, dirs, files in os.walk(output_directory):
            for file in files:
                id = uuid.uuid4().hex
                blob_name = f"{id}/{file}"
                url = upload_to_firebase_storage(os.path.join(root, file), blob_name)
                os.remove(os.path.join(root, file))
                message = {
                    "audio_url": url,
                    "id": id,
                    "content": analysis_results,
                    "chatId": chatId,
                    "sentAt": datetime.now(),
                    "isUser": False,
                    "role": "audio_analysis",
                }
                write_to_firestore(id, chatId, message)
                messages.append(message)

        return messages
    except Exception as e:
        print(f"Exception occurred: {str(e)}")
        traceback.print_exc()
        raise


@app.route('/v1/unmix', methods=['POST'])
def unmix_audio():
    if 'audio' not in request.files:
        return "No audio file in request", 400
    file = request.files['audio']
    chatId = request.form['chatId']

    if file.content_length > 50 * 1024 * 1024:
        return "File size exceeds limit of 50MB", 400

    temp_file = tempfile.NamedTemporaryFile(delete=False)
    file.save(temp_file.name)
    filepath = temp_file.name

    print(f"File saved at: {filepath}. File exists: {os.path.exists(filepath)}")

    output_directory = os.path.join('tmp', 'unmixed')
    create_output_directory(output_directory)

    try:
        unmixer = AudioUnmix()
        analysis_results = unmixer.run(filepath, output_directory)
        response = audio_to_message(analysis_results, output_directory, chatId)

        os.remove(filepath)
        return {"results": response}, 200
    except Exception as e:
        return str(e), 500


@app.route('/v1/unmix_youtube', methods=['POST'])
def unmix_youtube():
    if 'url' not in request.json:
        return "No url in request", 400
    url = request.json['url']
    chatId = request.json['chatId']

    output_directory = os.path.join('tmp', 'unmixed')
    create_output_directory(output_directory)

    try:
        unmixer = AudioUnmix()
        analysis_results = unmixer.run_from_youtube(url, output_directory)
        response = audio_to_message(analysis_results, output_directory, chatId)
        return {"results": response}, 200
    except Exception as e:
        return str(e), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=80)
