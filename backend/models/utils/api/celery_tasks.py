from celery import Celery
from config import settings

app = Celery('mediscan', broker=settings.REDIS_URL, backend=settings.REDIS_URL)

@app.task
def process_image(image_id):
    # Background processing for heavy tasks
    pass
