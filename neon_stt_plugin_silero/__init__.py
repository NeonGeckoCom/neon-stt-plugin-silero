from tempfile import gettempdir
from time import time
import torch
from ovos_plugin_manager.stt import STT


class ModelContainer:
    def __init__(self):
        self.models = {}

    def load_lang(self, lang):
        lang = lang.split("-")[0]
        self.models[lang] = torch.hub.load(
            repo_or_dir='snakers4/silero-models',
            model='silero_stt',
            language=lang)

    def unload_lang(self, lang):
        if lang in self.models:
            del self.models[lang]

    def predict_audio(self, audio, lang):
        # load audio TODO how to avoid saving to tmp file?
        path = f"{gettempdir()}/{time()}.wav"
        with open(path, "wb") as f:
            f.write(audio.get_wav_data())
        return self.predict_wav(path, lang)

    def predict_wav(self, path, lang):
        if lang not in self.models:
            self.load_lang(lang)

        _, _, utils = self.models[lang]
        (read_batch, split_into_batches,
         read_audio, prepare_model_input) = utils

        test_files = [path]
        batches = split_into_batches(test_files, batch_size=10)
        model_input = prepare_model_input(read_batch(batches[0]))
        return self.predict(model_input, lang)

    def predict(self, model_input, lang):
        if lang not in self.models:
            self.load_lang(lang)

        model, decoder, utils = self.models[lang]

        output = model(model_input)
        for example in output:
            return decoder(example.cpu())


class SileroSTT(STT):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.engine = ModelContainer()
        self.engine.load_lang(self.lang)

    def execute(self, audio, language=None):
        lang = language or self.lang
        return self.engine.predict_audio(audio, lang)


