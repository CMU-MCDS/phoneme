# Testing

1. Install persephone-web-API (https://github.com/persephone-tools/persephone-web-API):
```
git clone git@github.com:persephone-tools/persephone-web-API.git
virtualenv persephone_web_api_env
source persephone_web_api_env/bin/activate
cd persephone-web-API
pip install -r requirements.txt
```

2. Launch the persephone-web-API server:
```
python transcription_API_server.py
```

3. Launch user-interface server:
```
cd user_interface
python ui_server.py
```
