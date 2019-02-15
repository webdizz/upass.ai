import io
import picamera
import time
import numpy as np
from PIL import Image


def capture(size=(224, 224)):
    image_file = '/tmp/camera_%s.jpg' % time.strftime(
        '%d_%H_%S',
        time.gmtime()
    )

    with picamera.PiCamera() as camera:
        camera.exposure_mode = 'sports'
        camera.capture(
            image_file,
            resize=size,
            format='jpeg'
        )
    return image_file


def main():
    capture()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
