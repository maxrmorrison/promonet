import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from tqdm import tqdm
import string
import promonet


MODEL_ID = "openai/whisper-large-v3"

def from_audio(audio, sample_rate=None, gpu=None):
    device = f'cuda:{gpu}' if gpu is not None else 'cpu'

    if sample_rate is None: assert isinstance(audio, dict)

    if not hasattr(from_files_to_files, 'pipe'):

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        ).to(device)

        processor = AutoProcessor.from_pretrained(
            MODEL_ID
        )

        from_files_to_files.pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=128,
            chunk_length_s=30,
            batch_size=64,
            return_timestamps=False,
            torch_dtype=torch.float16,
            device=device,
        )

    # build sample rate dictionaries
    if isinstance(audio, list):
        assert sample_rate is not None
        audio = [
            {'sampling_rate': sample_rate, 'raw': a.to(torch.float32).squeeze(dim=0).cpu().numpy()}
            for a in audio
        ]
    elif isinstance(audio, torch.Tensor):
        assert sample_rate is not None
        audio = {'sampling_rate': sample_rate, 'raw': audio.to(torch.float32).squeeze(dim=0).cpu().numpy()}

    results = from_files_to_files.pipe(audio)

    if isinstance(results, dict):
        results = [results]

    output = []

    for result in results:
        if not isinstance(result['text'], str):
            result['text'] = str(result['text'])

        result_text = promonet.evaluate.metrics.normalize_text(result['text'].lower())
        output.append(result_text)
    if len(output) == 1:
        return output[0]
    return output

def from_files_to_files(audio_files, output_files, gpu=None):
    device = f'cuda:{gpu}' if gpu is not None else 'cpu'

    if not hasattr(from_files_to_files, 'pipe'):

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        ).to(device)

        processor = AutoProcessor.from_pretrained(
            MODEL_ID
        )

        from_files_to_files.pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=128,
            chunk_length_s=30,
            batch_size=64,
            return_timestamps=False,
            torch_dtype=torch.float16,
            device=device,
        )

    audio_files = [str(audio_file) for audio_file in audio_files]

    results = from_files_to_files.pipe(audio_files)

    for result, output_file in zip(results, output_files):
        with open(output_file, 'w', encoding='utf-8') as f:
            if not isinstance(result['text'], str):
                import pdb; pdb.set_trace()
                result['text'] = str(result['text'])

            result_text = promonet.evaluate.metrics.normalize_text(result['text'].lower())
            f.write(result_text)