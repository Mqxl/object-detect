import os
import shutil
import uuid
from io import BytesIO

from django.conf import settings
from django.shortcuts import render
from django.http import HttpResponse
from ultralytics import YOLO
from PIL import Image


def upload_photo(request):
    if request.method == 'POST' and request.FILES['photo']:
        # Обработка загруженной фотографии
        photo = request.FILES['photo']
        model = YOLO("yolov8m.pt")
        folder_name = str(uuid.uuid4())
        # Construct the path where the photo will be saved
        path = os.path.join(settings.BASE_DIR, folder_name, photo.name)

        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Save the uploaded photo at the specified path
        with open(path, 'wb') as f:
            for chunk in photo.chunks():
                f.write(chunk)
        results = model.predict(path)
        result = results[0]
        try:
            image = Image.fromarray(result.plot()[:,:,::-1])
            image_format = image.format or 'JPEG'
            image_bytes_io = BytesIO()
            image.save(image_bytes_io, format=image_format)
            image_bytes_io.seek(0)
            # Здесь вы можете добавить свой код для сохранения фотографии или выполнения других действий с ней
            shutil.rmtree(os.path.dirname(path))
            response = HttpResponse(image_bytes_io, content_type='image/jpeg')
            response['Content-Disposition'] = 'attachment; filename="processed_image.jpg"'
            return response
        except:
            shutil.rmtree(os.path.dirname(path))
            return HttpResponse("Error", status=400)
    return render(request, 'index.html')
