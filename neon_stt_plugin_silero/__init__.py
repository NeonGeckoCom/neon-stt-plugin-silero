from tempfile import gettempdir
from time import time
import onnx
import onnxruntime
import torch
from omegaconf import OmegaConf
from ovos_plugin_manager.templates.stt import STT


class SileroSTT(STT):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        lang = self.config.get("lang") or "en"
        self.lang = lang.split("-")[0]

        # load provided utils
        _, self.decoder, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-models',
            model='silero_stt',
            language=self.lang)
        (self.read_batch, self.split_into_batches,
         self.read_audio, self.prepare_model_input) = utils

        self.model = self.config.get("model")

        if not self.model:
            # see available models
            torch.hub.download_url_to_file(
                'https://raw.githubusercontent.com/snakers4/silero-models/master/models.yml',
                'models.yml')
            models = OmegaConf.load('models.yml')
            available_languages = list(models.stt_models.keys())
            if self.lang not in available_languages:
                raise ValueError(f"Invalid language! Provide your own model instead.\n"
                                 f"Only {list(models.stt_models.keys())} supported")
            torch.hub.download_url_to_file(models.stt_models.en.latest.onnx,
                                           'model.onnx', progress=True)

        # load the actual ONNX model
        onnx_model = onnx.load('model.onnx')
        onnx.checker.check_model(onnx_model)
        self.ort_session = onnxruntime.InferenceSession('model.onnx')

    def predict_audio(self, audio):
        # load audio TODO how to avoid saving to tmp file?
        path = f"{gettempdir()}/{time()}.wav"
        with open(path, "wb") as f:
            f.write(audio.get_wav_data())
        return self.predict_wav(path)

    def predict_wav(self, path):
        test_files = [path]
        batches = self.split_into_batches(test_files, batch_size=10)
        onnx_input = self.prepare_model_input(self.read_batch(batches[0]))
        return self.predict(onnx_input.detach().cpu().numpy())

    def predict(self, onnx_input):
        # actual onnx inference and decoding
        ort_inputs = {'input': onnx_input}
        ort_outs = self.ort_session.run(None, ort_inputs)
        return self.decoder(torch.Tensor(ort_outs[0])[0])

    def execute(self, audio, language=None):
        return self.predict_audio(audio)
