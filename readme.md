

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

TODO download to xdg path, check if exists before redownloading, add urls for manual download