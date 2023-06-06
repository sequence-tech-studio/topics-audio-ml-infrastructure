import torch
import torchaudio
import os
from openunmix import predict

class AudioUnmix:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def run(self, audio_file_path, out_dir):
        audio, rate = torchaudio.load(audio_file_path, normalize=True)
        audio = audio.to(self.device)
        estimates = predict.separate(
            torch.as_tensor(audio).float(),
            rate=rate,
            device=self.device
        )  
        # Write separated audio to files
        for target, estimate in estimates.items():
            estimate = estimate[0].cpu().detach().numpy()
            torchaudio.save(
                os.path.join(out_dir, f"{target}.wav"), 
                torch.tensor(estimate),
                sample_rate=rate
            )
