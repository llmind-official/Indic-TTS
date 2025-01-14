# 1. Create environment
git clone https://github.com/llmind-official/Indic-TTS.git
apt-get update --fix-missing
apt-get install libsndfile1-dev ffmpeg enchant -y

cd Indic-TTS
conda create -n tts-env
source activate tts-env

# 2. Setup PyTorch
pip install -U torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

# 3. Setup TTS
git clone https://github.com/llmind-official/TTS 
cd TTS
pip3 install -e .[all]
cd ..

# 5. Install other requirements
pip install -r requirements.txt

mkdir -p models/v1/
cd models/v1/
wget https://github.com/AI4Bharat/Indic-TTS/releases/download/v1-checkpoints-release/en.zip
wget https://github.com/AI4Bharat/Indic-TTS/releases/download/v1-checkpoints-release/hi.zip
unzip en.zip
unzip hi.zip
rm en.zip
rm hi.zip
cd ../../
cd inference/
mv ../models .
pip install -r requirements-server.txt
pip install -r requirements-utils.txt
uvicorn server:api --host "0.0.0.0" --port 6006
