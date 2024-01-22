from google.cloud import aiplatform

REGION = "us-central1"
BUCKET_NAME = "gs://my-bucket"

MODEL_LOC = 'gs://argolis-vertex-uscentral1/model_output_tr_en'
MODEL_DISPLAY_NAME="tr-en"
model = aiplatform.Model.upload(
    display_name=MODEL_DISPLAY_NAME,
    artifact_uri=MODEL_LOC,
    serving_container_image_uri=IMAGE_URI
)