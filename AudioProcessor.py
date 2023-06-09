import logging
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
            "bass": self._analyze_pitch_based,
            "drums": self.summarize_drums_analysis,
            "vocals": self._analyze_pitch_based,
            "other": self._analyze_pitch_based
        }
        self.samples = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.basicConfig(filename='audio_analysis.log', level=logging.INFO)
        logging.info('AudioAnalyzer initialized.')

    def analyze(self, target, audio_file_path):
        logging.info(f"Starting analysis. Target: {target}, File: {audio_file_path}")
        if target not in self.target_functions:
            logging.error(f"Unknown target: {target}")
            raise ValueError(f"Unknown target: {target}")
        
        if not os.path.exists(audio_file_path):
            logging.error(f"No such file or directory: {audio_file_path}")
            raise FileNotFoundError(f"No such file or directory: {audio_file_path}")

        self.samples = self._read_samples(audio_file_path)
        
        return self.target_functions[target](audio_file_path)
        

    def frequency_to_midi(self, frequency):
        return round(69 + 12 * np.log2(frequency / 440))

    def midi_to_note(self, midi):
        # mapping of MIDI note number to note names
        notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        octave = int(midi / 12) - 1
        note_idx = int(midi % 12)  # ensure note_idx is an integer
        return notes[note_idx] + str(octave)
    
    def _analyze_pitch_based(self, audio_file_path):
        analysis = self._analyze_general(audio_file_path)

        summary = self.create_summary(analysis["onset"], analysis["notes"])
        
        logging.info(f"Completed pitch-based analysis for {audio_file_path}")

        return summary

    def summarize_drums_analysis(self, audio_file_path):
        analysis = self._analyze_rhythm_based(audio_file_path)

        summary = self.create_summary(analysis["onset"], analysis["tempo"])
        
        logging.info(f"Completed rhythm-based analysis for {audio_file_path}")

        return summary

    def _analyze_general(self, audio_file_path):
        return {
            "pitch": self._pitch_analysis(audio_file_path),
            "onset": self._onset_analysis(audio_file_path),
            "notes": self._notes_analysis(audio_file_path),
        }

    def _analyze_rhythm_based(self, audio_file_path):
        return {
            "onset": self._onset_analysis(audio_file_path),
            "tempo": self._tempo_analysis(audio_file_path),
        }

    def create_summary(self, onset_analysis, notes_analysis):
        onset_summary = [f"{i*self.hop_size/self.samplerate:.2f}" for i, onset in enumerate(onset_analysis) if onset[0] > 0.5]
        notes_summary = [note for note in notes_analysis if len(note) > 1 and note[1] > 0.9]

        summary = [
            {
                "pitch": note[2],
                "length": onset,
                "note": self.midi_to_note(note[0])
            }
            for note, onset in zip(notes_summary, onset_summary)
        ]

        return summary
    
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
