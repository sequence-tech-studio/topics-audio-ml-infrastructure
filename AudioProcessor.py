from collections import Counter
import logging
import subprocess
import aubio
import torch
import torchaudio
import os
from openunmix import predict
from yt_dlp import YoutubeDL
import numpy as np

class AudioUnmix:
   
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def convert_to_pcm_wav(self,input_file, output_file):
        command = ['ffmpeg', '-i', input_file, '-f', 'wav', '-acodec', 'pcm_s16le', '-ar', '44100', output_file]
        subprocess.run(command, check=True)
    
    def analyze(self,file_path):
    # parameters
        downsample = 1
        samplerate = 44100 // downsample

        win_s = 512 // downsample  # fft size
        hop_s = 256 // downsample  # hop size
        converted_file_path = file_path + '.pcm.wav'
        self.convert_to_pcm_wav(file_path, converted_file_path)
        print(f"Analyzing {file_path}...")\
        
        s = aubio.source(converted_file_path, samplerate, hop_s)
        samplerate = s.samplerate

        tolerance = 0.4

        notes_o = aubio.notes("default", win_s, hop_s, samplerate)
        detected_notes = []
        

        # total number of frames read
        total_frames = 0
        while True:
            samples, read = s()
            new_note = notes_o(samples)
            velocity = new_note[2] / 127.  # scale velocity from 0 to 1
            if new_note[0] != 0 and velocity > tolerance:  # if a new note was detected
                note_str = aubio.midi2note(int(new_note[1] if new_note[1] <=127 else 127))
                detected_notes.append({
                            'start': float(new_note[0]),
                            'note': note_str,
                            'velocity': velocity
                            })
            total_frames += read
            if read < hop_s: break
        os.remove(converted_file_path)
        return detected_notes
    
    def drums_analysis(self, file_path, win_s=512, hop_s=256):
        samplerate = 0  # let aubio automatically determine the samplerate
        converted_file_path = file_path + '.pcm.wav'
        self.convert_to_pcm_wav(file_path, converted_file_path)
        s = aubio.source(converted_file_path, samplerate, hop_s)
        

        # create a 'tempo' object that stores and computes BPM
        o = aubio.tempo("default", win_s, hop_s, samplerate)

        bpm_estimates = []
        total_frames = 0

        while True:
            # read from audio file
            samples, read = s()

            # calculate bpm for current frame
            is_beat = o(samples)
            if is_beat:
                this_bpm = o.get_bpm()
                if this_bpm > 0:  # check if valid BPM is calculated
                    bpm_estimates.append(this_bpm)

            # increment total frames
            total_frames += read
            if read < hop_s:
                break

        # Calculate average BPM
        average_bpm = sum(bpm_estimates) / len(bpm_estimates) if bpm_estimates else 0

        return {"bpm": average_bpm}

    def run(self, audio_file_path, out_dir):
        audio, rate = torchaudio.load(audio_file_path)
        audio = audio.to(self.device)
        estimates = predict.separate(
            torch.as_tensor(audio).float(),
            rate=rate,
            device=self.device
        )  
        analysis_results = {}
        # Write separated audio to files
        for target, estimate in estimates.items():
            estimate = estimate[0].cpu().detach().numpy()
            file_path = os.path.join(out_dir, f"{target}.wav")
            torchaudio.save(
                file_path, 
                torch.tensor(estimate),
                sample_rate=rate
            )
            # Analyze the separated audio
            if target != 'drums':
                analysis_results[target] = self.analyze(file_path)
            # else:
            #     analysis_results[target] = self.drums_analysis(file_path)

        return analysis_results
            
    def run_from_youtube(self, url, out_dir):   
        print("url:", url)  # Add this line
        print("out_dir:", out_dir)  # Add this line
        print("Downloading audio from youtube...")
        print(out_dir)
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
        }],
        'outtmpl': os.path.join(out_dir, 'temp'),
        }
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        self.run(os.path.join(out_dir, 'temp.wav'), out_dir) 
