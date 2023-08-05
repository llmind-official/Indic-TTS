python3 -m TTS.bin.synthesize --text "Delhi is a beautiful city. It is capital of India." \
    --model_path models/v1/en/fastpitch/best_model.pth \
    --config_path models/v1/en/fastpitch/config.json \
    --vocoder_path models/v1/en/hifigan/best_model.pth \
    --vocoder_config_path models/v1/en/hifigan/config.json \
    --out_path ../out.wav
