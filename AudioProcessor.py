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
        self.analyzer = AudioAnalyzer()

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
            analysis_results[target] = self.analyzer.analyze(target, file_path)

        return analysis_results
            
    def run_from_youtube(self, url, out_dir):
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
            }],
            'outtmpl': 'tmp/temp',
        }
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        self.run('tmp/temp.wav', out_dir)

class AudioAnalyzer:
    def __init__(self, buf_size=1024, hop_size=512, samplerate=44100):
        self.buf_size = buf_size
        self.hop_size = hop_size
        self.samplerate = samplerate
        self.target_functions = {
            "bass": self.analyze_bass,
            "drums": self.analyze_drums,
            "vocals": self.analyze_vocals,
            "other": self.analyze_others
        }
        self.samples = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def analyze(self, target, audio_file_path):
        if target in self.target_functions:
            self.samples = self._read_samples(audio_file_path)
            
            return self.target_functions[target](audio_file_path)
        else:
            raise ValueError(f"Unknown target: {target}")
        
    def analyze_bass(self, audio_file_path):
        return {
            "pitch": self._pitch_analysis(audio_file_path),
            "onset": self._onset_analysis(audio_file_path),
        }

    def analyze_drums(self, audio_file_path):
        return {
            "onset": self._onset_analysis(audio_file_path),
            "tempo": self._tempo_analysis(audio_file_path),
        }

    def analyze_vocals(self, audio_file_path):
        return {
            "pitch": self._pitch_analysis(audio_file_path),
            "onset": self._onset_analysis(audio_file_path),
            "notes": self._notes_analysis(audio_file_path),
        }

    def analyze_others(self, audio_file_path):
        return {
            "pitch": self._pitch_analysis(audio_file_path),
            "onset": self._onset_analysis(audio_file_path),
            "notes": self._notes_analysis(audio_file_path),
        }

    def _onset_analysis(self, audio_file_path):
        onset = aubio.onset("default", self.buf_size, self.hop_size, self.samplerate)
        samples = self._read_samples(audio_file_path)
        return [onset(samples[i:i+self.hop_size]).tolist() for i in range(0, len(samples), self.hop_size)]

    def _pitch_analysis(self, audio_file_path, method="default"):
        pitch = aubio.pitch(method, self.buf_size, self.hop_size, self.samplerate)
        samples = self._read_samples(audio_file_path)
        return [pitch(samples[i:i+self.hop_size]).tolist() for i in range(0, len(samples), self.hop_size)]

    def _tempo_analysis(self, audio_file_path):
        tempo = aubio.tempo("default", self.buf_size, self.hop_size, self.samplerate)
        samples = self._read_samples(audio_file_path)
        return [tempo(samples[i:i+self.hop_size]).tolist() for i in range(0, len(samples), self.hop_size)]

    def _notes_analysis(self, audio_file_path):
        notes = aubio.notes("default", self.buf_size, self.hop_size, self.samplerate)
        samples = self._read_samples(audio_file_path)
        return [notes(samples[i:i+self.hop_size]).tolist() for i in range(0, len(samples), self.hop_size)]

    def _read_samples(self, audio_file_path):
        if self.samples is None:
            audio, _ = torchaudio.load(audio_file_path)
            audio = audio.to(self.device)
            self.samples = audio[0].cpu().numpy()
            self.samples = np.pad(self.samples, (0, self.hop_size - len(self.samples) % self.hop_size), mode='constant')
        return self.samples