""" Translate from Turkish to English using the Google Translate API (basic tier)
    Requires authentication to Google Cloud Services
"""

import six
from google.cloud import translate_v2 as translate

def translate_text(target, text):


    translate_client = translate.Client()

    if isinstance(text, six.binary_type):
        text = text.decode("utf-8")

    # Text can also be a sequence of strings, in which case this method
    # will return a sequence of results for each text.
    result = translate_client.translate(text, target_language=target)

    print(u"Text: {}".format(result["input"]))
    print(u"Translation: {}".format(result["translatedText"]))
    print(u"Detected source language: {}".format(result["detectedSourceLanguage"]))

translate_text("en", "Ceplerini kontrol et.")
translate_text("en", "Ben Kuzey Koreliyim.")