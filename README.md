![Image](assets/resources/vippdiism.png)

# SIFT-based-copy-move-forgery-detection
SIFT based copy move forgery detection and localisation

## Copy-move detection

Detect copy-move into an image:
~~~
im = cv2.imread("assets/fiori-gialli.jpg")
score = copy_move_detector(im)
~~~

![input image](assets/fiori-gialli.jpg)
![detection](assets/output/DETECTION_fiori_gialli.png)

## Copy-move localisation

Localize copy-move into an image:
~~~
image = cv2.imread('./assets/fiori-gialli.jpg')
    mask, score, _ = copy_move_localisation(image)
    bin_mask = 255 * (np.uint8(mask > 0))
~~~

![input image](assets/fiori-gialli.jpg)
![mask](assets/output/MASK_fiori_gialli.png)
![localisation](assets/output/LOCALISATION_fiori_gialli.png)`

## Parameter configuration

To tweak algorithm parameters edit the values in _configuration.py_.