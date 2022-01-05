# NEON AI (TM) SOFTWARE, Software Development Kit & Application Development System
# All trademark and other rights reserved by their respective owners
# Copyright 2008-2021 Neongecko.com Inc.
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
# following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions
#    and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions
#    and the following disclaimer in the documentation and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote
#    products derived from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
# USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


from neon_stt_plugin_silero import SileroSTT
from ovos_utils.log import LOG
import unittest
import os




ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
TEST_PATH_DE = os.path.join(ROOT_DIR, "test_audio/de")
TEST_PATH_EN = os.path.join(ROOT_DIR, "test_audio/en")
TEST_PATH_UA = os.path.join(ROOT_DIR, "test_audio/ua")


class TestGetSTT(unittest.TestCase):
    def setUp(self) -> None:
        self.stt = SileroSTT('ua')

    def test_en_stt(self):
        LOG.info("ENGLISH STT MODEL")
        self.stt = SileroSTT('en')
        for file in os.listdir(TEST_PATH_EN):
            path = ROOT_DIR+'/test_audio/en/'+file
            text = self.stt.predict_wav(path)
            print(text)

    def test_de_stt(self):
        LOG.info("GERMAN STT MODEL")
        self.stt = SileroSTT('de')
        for file in os.listdir(TEST_PATH_DE):
            path = ROOT_DIR+'/test_audio/de/'+file
            text = self.stt.predict_wav(path)
            print(text)

    def test_ua_stt(self):
        LOG.info("UKRAINIAN STT MODEL")
        self.stt = SileroSTT('ua')
        for file in os.listdir(TEST_PATH_UA):
            path = ROOT_DIR+'/test_audio/ua/'+file
            text = self.stt.predict_wav(path)
            print(text)

if __name__ == '__main__':
    unittest.main()
