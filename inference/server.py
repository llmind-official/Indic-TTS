import io
import base64
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from TTS.utils.synthesizer import Synthesizer

from src.inference import TextToSpeechEngine
from src.models.request import TTSRequest, Sentence, TTSConfig
from src.models.response import TTSReponse
from fastapi.responses import StreamingResponse

from scipy.io.wavfile import write as scipy_wav_write
SUPPORTED_LANGUAGES = {
    'as' : "Assamese - অসমীয়া",
    'bn' : "Bangla - বাংলা",
    'brx': "Boro - बड़ो",
    'en' : "English (Indian accent)",
    'en+hi': "English+Hindi (Hinglish code-mixed)",
    'gu' : "Gujarati - ગુજરાતી",
    'hi' : "Hindi - हिंदी",
    'kn' : "Kannada - ಕನ್ನಡ",
    'ml' : "Malayalam - മലയാളം",
    'mni': "Manipuri - মিতৈলোন",
    'mr' : "Marathi - मराठी",
    'or' : "Oriya - ଓଡ଼ିଆ",
    'pa' : "Panjabi - ਪੰਜਾਬੀ",
    'raj': "Rajasthani - राजस्थानी",
    'ta' : "Tamil - தமிழ்",
    'te' : "Telugu - తెలుగు",
}

models = {}
for lang in SUPPORTED_LANGUAGES:
    try:
      models[lang]  = Synthesizer(
          tts_checkpoint=f'model/v1/{lang}/fastpitch/best_model.pth',
          tts_config_path=f'model/v1/{lang}/fastpitch/config.json',
          tts_speakers_file=f'model/v1/{lang}/fastpitch/speakers.pth',
          # tts_speakers_file=None,
          tts_languages_file=None,
          vocoder_checkpoint=f'model/v1/{lang}/hifigan/best_model.pth',
          vocoder_config=f'model/v1/{lang}/hifigan/config.json',
          encoder_checkpoint="",
          encoder_config="",
          use_cuda=True,
      )
    except:
      print(f"Skipping... {lang}")
    print(f"Synthesizer loaded for {lang}.")
    print("*"*100)

engine = TextToSpeechEngine(models)

api = FastAPI()

@api.get("/supported_languages")
def get_supported_languages():
    return SUPPORTED_LANGUAGES

@api.get("/")
def homepage():
    return "AI4Bharat Text-To-Speech API"

@api.post("/")
async def batch_tts(request: TTSRequest):
    print(request.items())

@api.post("/tts")
async def batch_tts(request: Request):
    body = await request.json()
    print(body)
    text = body['input'][0]
    lang = body['config']['language']
    speaker = body['config']['gender']
    #request = TTSRequest(input=list(map(lambda i: Sentence(source=i), body['input'])), config=TTSConfig(language=Language(sourceLanguage=body['config']['language']), gender=body['config']['gender']))
    ret = engine.infer_from_text(text, lang, speaker)
    byte_io = io.BytesIO()
    scipy_wav_write(byte_io, 22050, ret)
    encoded_bytes = base64.b64encode(byte_io.read())
    return Response(content=encoded_bytes, media_type="audio/mpeg")

#@api.post("/tts")
#async def batch_tts(request: Request, response: TTSResponse):
#    body = await request.json()
#    print(body)
#    text = body['input'][0]
#    lang = body['config']['language']
#    speaker = body['config']['gender']
#    request = TTSRequest(input=list(map(lambda i: Sentence(source=i), body['input'])), config=TTSConfig(language=Language(sourceLanguage=body['config']['language']), gender=body['config']['gender']))
#    resp = engine.infer_from_request(request)
#    #ret = engine.infer_from_text(text, lang, speaker)
#    #byte_io = io.BytesIO()
#    #scipy_wav_write(byte_io, 22050, ret)
#    #encoded_bytes = base64.b64encode(byte_io.read())
#    #encoded_string = encoded_bytes.decode()
#    #return StreamingResponse(encoded_string, media_type="audio/mpeg")
#    return resp


if __name__ == "__main__":
    # uvicorn server:api --host 0.0.0.0 --port 5050 --log-level info
    uvicorn.run("server:api", host="0.0.0.0", port=6006, log_level="info")
