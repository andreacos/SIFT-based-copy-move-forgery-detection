import os
import logging
import argparse
import numpy as np
from time import time
from datetime import datetime

import cv2

from localisation import copy_move_localisation
from utils import read_image


logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')
log = logging.getLogger(__name__)


def check_args(cmd_args):
    """ Check arguments' values
    :param cmd_args: command line arguments
    :return: True if parameters are correct False otherwise
    """
    if not os.path.exists(cmd_args.image):
        log.error("ERROR: Cannot find image {}".format(cmd_args.image))
        return False
    return True


def run(cmd_args):

    mask_path = os.path.join(os.path.dirname(cmd_args.image),
                             'mask-' + os.path.splitext(os.path.basename(cmd_args.image))[0] + '.png')

    if cmd_args.verbose == 1:
        log.info(args)
        log.info(f'Starting analysis')
        log.info(f'Input image {cmd_args.image}')
        log.info(f'Image type: {os.path.splitext(cmd_args.image)[-1]}')
        log.info(f'Output mask: {mask_path}')
        log.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    start = time()

    try:

        if os.path.splitext(cmd_args.image)[-1].lower() == '.gif':
            log.warning('The detector can not process GIF images. Exiting.')

        else:
            # Read image, as grayscale
            img = read_image(cmd_args.image)

            if img is None:
                log.error('Image does not exist or format is not supported. Exiting.')
                return
            else:
                mask, score, _ = copy_move_localisation(img)
                cv2.imwrite(mask_path, 255 * (np.uint8(mask > 0)))

                if cmd_args.verbose == 1:
                    log.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    log.info(f"Elapsed: {time() - start:3.2f}")
                    log.info(f"Score (number of matches): {score}")

    except Exception as ex:
        log.error(f'{str(ex)}')
        raise ex

    return


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="SIFT-based Copy-move detection CLI version")
    parser.add_argument("image", type=str, help="path to the input image (all formats supported by OpenCV)")
    parser.add_argument("-v", "--verbose", type=int, choices=[0, 1], default=1,
                        help="Modify program's verbosity (0 silent, 1 verbose)")

    # Parse command line
    args = parser.parse_args()

    if check_args(args):
        run(args)

