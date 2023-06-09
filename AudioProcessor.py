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
            "bass": self.summarize_bass_analysis,
            "drums": self.summarize_drums_analysis,
            "vocals": self.summarize_vocals_analysis,
            "other": self.summarize_others_analysis
        }
        self.samples = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def analyze(self, target, audio_file_path):
        if target in self.target_functions:
            self.samples = self._read_samples(audio_file_path)
            
            return self.target_functions[target](audio_file_path)
        else:
            raise ValueError(f"Unknown target: {target}")
        

    def frequency_to_midi(self, frequency):
        return round(69 + 12 * np.log2(frequency / 440))

    def midi_to_note(self, midi):
        # mapping of MIDI note number to note names
        notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        octave = int(midi / 12) - 1
        note_idx = int(midi % 12)  # ensure note_idx is an integer
        return notes[note_idx] + str(octave)
    
    def summarize_bass_analysis(self, audio_file_path):
        analysis = self.analyze_bass(audio_file_path)

        pitch_analysis = analysis["pitch"]
        onset_analysis = analysis["onset"]

        pitch_summary = [self.midi_to_note(self.frequency_to_midi(pitch[0])) for pitch in pitch_analysis if pitch[0] > 0]
        onset_summary = [f"{i*self.hop_size/self.samplerate:.2f} seconds" for i, onset in enumerate(onset_analysis) if onset[0] > 0.5]

        return {
            "pitch": " -> ".join(pitch_summary),
            "onset": ", ".join(onset_summary)
        }

    def summarize_drums_analysis(self, audio_file_path):
        analysis = self.analyze_drums(audio_file_path)

        onset_analysis = analysis["onset"]
        tempo_analysis = analysis["tempo"]

        onset_summary = [f"{i*self.hop_size/self.samplerate:.2f} seconds" for i, onset in enumerate(onset_analysis) if onset[0] > 0.5]
        tempo_summary = [f"Beat at {i*self.hop_size/self.samplerate:.2f} seconds" for i, tempo in enumerate(tempo_analysis) if tempo[0] > 0]

        return {
            "onset": ", ".join(onset_summary),
            "tempo": ", ".join(tempo_summary)
        }

    def summarize_vocals_analysis(self, audio_file_path):
        analysis = self.analyze_vocals(audio_file_path)

        pitch_analysis = analysis["pitch"]
        onset_analysis = analysis["onset"]
        notes_analysis = analysis["notes"]

        pitch_summary = [self.midi_to_note(self.frequency_to_midi(pitch[0])) for pitch in pitch_analysis if pitch[0] > 0]
        onset_summary = [f"{i*self.hop_size/self.samplerate:.2f} seconds" for i, onset in enumerate(onset_analysis) if onset[0] > 0.5]
        notes_summary = [self.midi_to_note(note[0]) for note in notes_analysis if note[0] > 0]

        return {
            "pitch": " -> ".join(pitch_summary),
            "onset": ", ".join(onset_summary),
            "notes": ", ".join(notes_summary)
        }

    def summarize_others_analysis(self, audio_file_path):
        analysis = self.analyze_others(audio_file_path)

        pitch_analysis = analysis["pitch"]
        onset_analysis = analysis["onset"]
        notes_analysis = analysis["notes"]

        pitch_summary = [self.midi_to_note(self.frequency_to_midi(pitch[0])) for pitch in pitch_analysis if pitch[0] > 0]
        onset_summary = [f"{i*self.hop_size/self.samplerate:.2f} seconds" for i, onset in enumerate(onset_analysis) if onset[0] > 0.5]
        notes_summary = [self.midi_to_note(note[0]) for note in notes_analysis if note[0] > 0]

        return {
            "pitch": " -> ".join(pitch_summary),
            "onset": ", ".join(onset_summary),
            "notes": ", ".join(notes_summary)
        }
    
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