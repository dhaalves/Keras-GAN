from PIL import Image
from keras.engine.saving import load_model
import numpy as np


class BaseGAN():
    def __init__(self, images_folder, epochs, batch_size):
        self.batch_size = batch_size
        self.epochs = epochs


def normalize(gen_imgs):
    # Normalised [0,1]
    gen_imgs = (gen_imgs - np.min(gen_imgs)) / np.ptp(gen_imgs)

    # Normalised [0,255] as integer
    gen_imgs = 255 * (gen_imgs - np.min(gen_imgs)) / np.ptp(gen_imgs).astype(int)
    return gen_imgs.astype(np.uint8)


def generate_img(filepath, size=1):
    model = load_model(filepath)
    noises = np.random.normal(0, 1, (size, 100))
    imgs_gen = model.predict(noises)
    for i, img in enumerate(imgs_gen):
        img = normalize(img)
        # img_norm = cv2.normalize(img_gen, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        # img_norm = img_norm.astype(np.uint8)
        # img_norm_rgb = cv2.cvtColor(img_norm, cv2.COLOR_BGR2RGB)
        # cv2.imwrite('color_img_norm_cv.jpg', img_norm_rgb)
        Image.fromarray(img, 'RGB').save('img_%d.png' % i)
