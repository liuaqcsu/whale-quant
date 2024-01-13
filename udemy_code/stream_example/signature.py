import hmac
import hashlib
import base64

WEB_ID = "0100e921-f640-410d-90c6-efaaa18abeb7"
WEB_KEY = "k5Qxfb6jyxNkW4cp"
SECRET = "d3mXrQfrhfCBjq7y3j5Phzm85rtYPphfjpzfdX4wcqghXfbq4mJR7wgNNn39wsM4"


def get_signature(timestamp):
    fullsec = timestamp + WEB_ID + WEB_KEY
    
    msg = fullsec.encode('utf-8')
    secret = SECRET.encode('utf-8')

    hashed = hmac.new(secret, msg, hashlib.sha256).digest()
    encoded_string = base64.b64encode(hashed)

    return encoded_string.decode('utf-8')
