import pyaudio
import dashscope
from dashscope.audio.tts_v2 import *

dashscope.api_key = 'xxxxxxxxxxxxxxxxxx'


class Callback(ResultCallback):
    _player = None
    _stream = None

    def on_open(self):
        print("websocket is open.")
        self._player = pyaudio.PyAudio()
        self._stream = self._player.open(
            format=pyaudio.paInt16, channels=1, rate=16000, output=True
        )

    def on_complete(self):
        print("speech synthesis task complete successfully.")

    def on_error(self, message: str):
        print(f"speech synthesis task failed, {message}")

    def on_close(self):
        print("websocket is closed.")
        # 停止播放器
        self._stream.stop_stream()
        self._stream.close()
        self._player.terminate()

    def on_event(self, message):
        print(f"recv speech synthsis message {message}")

    def on_data(self, data: bytes) -> None:
        print("audio result length:", len(data))
        self._stream.write(data)


if __name__ == '__main__':
    synthesizer = SpeechSynthesizer(
        model="cosyvoice-v1",
        voice="longxiaochun",
        format=AudioFormat.WAV_16000HZ_MONO_16BIT
    )
    audio = synthesizer.call('已到达目的地')
    with open('position/arrival.wav', 'wb') as f:
        f.write(audio)