# Import required libraries.

print("Loading the required Python modules for the ResNet152 model.")

from objreg_utils import (
    to_img,
    img_url_to_json,
    plot_predictions,
    plot_single_prediction,
    get_model_api,
    tab_complete_path,
    validateURL,
)

import glob
import json
import os
import sys
import readline

# The working dir of the command which invokes this script.
CMD_CWD=os.environ.get('CMD_CWD', '')

# Utilities

def _score_for_one_img(img, label='image'):
    """Score for a single image in url.

    Args:
        url (str): a url to an image, or a path to an image.
    """

    jsonimg = img_url_to_json(img, label=label)
    json_lod= json.loads(jsonimg)
    output = predict_for(json_lod)
    plot_single_prediction(img, output)
    return output


def _score_for(url):
    """Score for the images in url.

    Args:
        url (str): a url to an image, or a path to an image, or a dir for images.
    """

    if validateURL(url):
        _score_for_one_img(url, label=url)
    else:
        # Change to the dir of command which invokes this script
        if CMD_CWD != '':
            oldwd = os.getcwd()
            os.chdir(CMD_CWD)

        url = os.path.abspath(os.path.expanduser(url))

        if CMD_CWD != '':
            os.chdir(oldwd)

        if os.path.isdir(url):
            for img in os.listdir(url):
                img_file = os.path.join(url, '', img)
                _score_for_one_img(img_file, label=img_file)
        else:
            _score_for_one_img(url, label=url)


# Load model

predict_for = get_model_api()


# Setup input path completion

readline.set_completer_delims('\t')
readline.parse_and_bind("tab: complete")
readline.set_completer(tab_complete_path)

# Scoring

if len(sys.argv) < 2:
    try:
        url = input('\nPath or URL of images to recognize:\n> ')
    except EOFError:
        print()
        sys.exit(0)

    while url != '':
        _score_for(url)

        try:
            url = input('\nPath or URL of images to recognize:\n> ')
        except EOFError:
            print()
            sys.exit(0)
else:
    for url in sys.argv[1:]:
        _score_for(url)
