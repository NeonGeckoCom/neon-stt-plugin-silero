

# config

```json

{
  "stt": {
     "module": "neon-stt-plugin-silero",
     "neon-stt-plugin-silero": {
        "model": "model.onnx"
     }
  }
}
```

model is optional, will be downloaded at runtime based on lang if missing


## Docker

This plugin can be used together with [ovos-stt-http-server](https://github.com/OpenVoiceOS/ovos-stt-http-server) 

```bash
docker run -p 8080:8080 ghcr.io/neongeckocom/silero-stt:master
```