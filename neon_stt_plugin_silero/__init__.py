from tempfile import gettempdir
from time import time
import torch
from ovos_plugin_manager.stt import STT


class SileroSTT(STT):
    def __init__(self, lang, *args, **kwargs):
        super().__init__(*args, **kwargs)
        lang = lang or "en"
        self.lang = lang.split("-")[0]

        # load provided utils
        self.model, self.decoder, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-models',
            model='silero_stt',
            language=self.lang)
        (self.read_batch, self.split_into_batches,
         self.read_audio, self.prepare_model_input) = utils

    def predict_audio(self, audio):
        # load audio TODO how to avoid saving to tmp file?
        path = f"{gettempdir()}/{time()}.wav"
        with open(path, "wb") as f:
            f.write(audio.get_wav_data())
        return self.predict_wav(path)

    def predict_wav(self, path):
        test_files = [path]
        batches = self.split_into_batches(test_files, batch_size=10)
        model_input = self.prepare_model_input(self.read_batch(batches[0]))
        return self.predict(model_input)

    def predict(self, model_input):
        output = self.model(model_input)
        for example in output:
            return self.decoder(example.cpu())

    def execute(self, audio, language=None):
        return self.predict_audio(audio)


