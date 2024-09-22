
import torch
import torch.nn.functional as F

class AudioProcessor(torch.nn.Module):
    def __init__(self, n_fft, hop_length, win_length, sample_rate, mel_bins, 
                 f_min, f_max, normalization_mode="per_channel",
                 int8_mode=False, bf16_mode=False):
        super(AudioProcessor, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sample_rate = sample_rate
        self.mel_bins = mel_bins
        self.f_min = f_min
        self.f_max = f_max
        self.normalization_mode = normalization_mode
        self.int8_mode = int8_mode
        self.bf16_mode = bf16_mode

        # Define mel filterbank
        self.mel_filterbank = torch.nn.functional.mel_scale(
            n_mels=self.mel_bins,
            f_min=self.f_min,
            f_max=self.f_max,
            n_stft=self.n_fft // 2 + 1,
            sample_rate=self.sample_rate
        )

    def forward(self, audio_data):
        """
        Processes audio data through the following steps:
        - STFT
        - Mel spectrogram
        - Normalization (per-channel or global)
        - Cross-fade
        - Adaptive average pooling
        """
        if self.bf16_mode:
            audio_data = audio_data.to(torch.bfloat16)
        else:
            if self.int8_mode:
                audio_data = audio_data.to(torch.int8)

        # STFT
        stft = torch.stft(
            audio_data,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=torch.hamming_window(self.win_length)
        )

        # Magnitude and Mel Spectrogram
        magnitude = torch.abs(stft)
        mel_spectrogram = torch.matmul(magnitude, self.mel_filterbank)

        # Normalization
        if self.normalization_mode == "per_channel":
            mel_spectrogram = F.layer_norm(mel_spectrogram, (mel_spectrogram.shape[1],))
        elif self.normalization_mode == "global":
            mel_spectrogram = F.layer_norm(mel_spectrogram, (mel_spectrogram.shape[1], mel_spectrogram.shape[2]))
        
        # Cross-fade
        fade_in_length = int(0.1 * self.sample_rate / self.hop_length)  # 100ms fade-in
        fade_out_length = int(0.1 * self.sample_rate / self.hop_length)  # 100ms fade-out
        mel_spectrogram = F.cross_fade(mel_spectrogram, mel_spectrogram, fade_in_length, fade_out_length)

        # Adaptive Average Pooling
        mel_spectrogram = F.adaptive_avg_pool1d(mel_spectrogram, 1)

        if self.bf16_mode:
            return mel_spectrogram.to(torch.float32)
        else:
            if self.int8_mode:
                return mel_spectrogram.to(torch.float32)
            else:
                return mel_spectrogram

function_signature = {
    "name": "audio_processor",
    "inputs": [
        ((1, 16000), torch.float32),
    ],
    "outputs": [
        ((1, 128, 1), torch.float32)
    ]
}
