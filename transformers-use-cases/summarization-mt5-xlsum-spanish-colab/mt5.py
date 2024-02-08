""" Simple mT5 inference using AutoTokenizer, AutoModelForSeq2SeqLM classes
    Added to summarization notebook
"""

import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

WHITESPACE_HANDLER = lambda k: re.sub('\s+', ' ', re.sub('\n+', ' ', k.strip()))

article_text = """La Guardia Civil ha desarticulado un grupo organizado dedicado a copiar en los examenes teoricos 
            para la obtencion del permiso de conducir. Para ello, empleaban receptores y camaras de alta tecnologia 
            y operaban desde la misma sede del Centro de examenes de la Direccion General de Trafico (DGT) en 
            Mostoles. Es lo que han llamado la Operacion pinga. El grupo desarticulado ofrecia el servicio de 
            transporte y tecnologia para copiar y poder aprobar. Por dicho servicio cobraban 1.000 euros. 
            Los investigadores sorprendieron in fraganti a una mujer intentando copiar en el examen. 
            Portaba una chaqueta con dispositivos electronicos ocultos, concretamente un telefono movil al que 
            estaba conectada una camara que habia sido insertada en la parte frontal de la chaqueta para transmitir 
            online el examen y que orientada al ordenador del Centro de Examenes en el que aparecen las preguntas, 
            permitia visualizar las imagenes en otro ordenador alojado en el interior de un vehiculo estacionado en 
            las inmediaciones del centro. En este vehiculo, se encontraban el resto del grupo desarticulado con 
            varios ordenadores portatiles y tablets abiertos y conectados a paginas de test de la DGT para consultar 
            las respuestas. 
            Estos, comunicaban con la mujer que estaba en el aula haciendo el examen a traves de un 
            diminuto receptor bluetooth que portaba en el interior de su oido.  
            Luis de Lama, portavoz de la Guardia Civil de Trafico destaca que los ciudadanos, eran de origen chino,     
            y copiaban en el examen utilizando la tecnologia facilitada por una organizacion. 
            Destaca que, ademas de parte del fraude que supone copiar en un examen muchos de estos ciudadanos desconocian el idioma, 
            no hablan ni entienden el español lo que supone un grave riesgo para la seguridad vial por desconocer 
            las señales y letreros que avisan en carretera de muchas incidencias. """
            
model_name = "csebuetnlp/mT5_multilingual_XLSum"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

input_ids = tokenizer(
    article_text, #[WHITESPACE_HANDLER(article_text)],
    return_tensors="pt", #mandatory 
    padding="max_length",
    truncation=True,
    max_length=512
)["input_ids"]

output_ids = model.generate(
    input_ids=input_ids,
    max_length=84,
    no_repeat_ngram_size=2,
    num_beams=4
)[0]

summary = tokenizer.decode(
    output_ids,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False
)

print(summary)