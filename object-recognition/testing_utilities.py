import base64
import json
import urllib
from io import BytesIO

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import toolz
from PIL import Image, ImageOps
import random
import re

urlregex = re.compile(
           r'^(?:http|ftp)s?://' # http:// or https://
           r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|' #domain...
           r'localhost|' #localhost...
           r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})' # ...or ip
           r'(?::\d+)?' # optional port
           r'(?:/?|[/?]\S+)$', re.IGNORECASE)


def read_image_from(url):
    return toolz.pipe(
        url,
        urllib.request.urlopen if re.match(urlregex, url) is not None else lambda x: open(x, 'rb'),
        lambda x: x.read(),
        BytesIO)


def to_rgb(img_bytes):
    return Image.open(img_bytes).convert('RGB')


@toolz.curry
def resize(img_file, new_size=(100, 100)):
    return ImageOps.fit(img_file, new_size, Image.ANTIALIAS)


def to_base64(img):
    imgio = BytesIO()
    img.save(imgio, 'PNG')
    imgio.seek(0)
    dataimg = base64.b64encode(imgio.read())
    return dataimg.decode('utf-8')


def to_img(img_url):
    return toolz.pipe(img_url,
                      read_image_from,
                      to_rgb,
                      resize(new_size=(224,224)))


def img_url_to_json(url, label='image'):
    img_data = toolz.pipe(url,
                          to_img,
                          to_base64)
    return json.dumps({label:'\"{0}\"'.format(img_data)})


def  _plot_image(ax, img):
    ax.imshow(to_img(img))
    ax.tick_params(axis='both',       
                   which='both',      
                   bottom='off',      
                   top='off',         
                   left='off',
                   right='off',
                   labelleft='off',
                   labelbottom='off') 
    return ax


def _plot_prediction_bar(ax, r):
    res = list(r[0].values())[0]
    perf = list(c[1] for c in res)
    ax.barh(range(3, 0, -1), perf, align='center', color='#55DD55')
    ax.tick_params(axis='both',       
                   which='both',      
                   bottom='off',      
                   top='off',         
                   left='off',
                   right='off',
                   labelbottom='off') 
    tick_labels = reversed(list(' '.join(c[0].split()[1:]).split(',')[0] for c in res))
    ax.yaxis.set_ticks([1,2,3])
    ax.yaxis.set_ticklabels(tick_labels, position=(0.5,0), minor=False, horizontalalignment='center')

    
def plot_predictions(images, classification_results):
    if len(images)!=6:
        raise Exception('This method is only designed for 6 images')
    gs = gridspec.GridSpec(2, 3)
    fig = plt.figure(figsize=(12, 9))
    gs.update(hspace=0.1, wspace=0.001)
    
    for gg, r, img in zip(gs, classification_results, images):
        gg2 = gridspec.GridSpecFromSubplotSpec(4, 10, subplot_spec=gg)
        ax = fig.add_subplot(gg2[0:3, :])
        _plot_image(ax, img)
        ax = fig.add_subplot(gg2[3, 1:9])
        _plot_prediction_bar(ax, r)

    plt.show()
        
def write_json_to_file(json_dict, filename, mode='w'):
    with open(filename, mode) as outfile:
        json.dump(json_dict, outfile, indent=4,sort_keys=True)
        outfile.write('\n\n')
        
def gen_variations_of_one_image(IMAGEURL, num, label='image'):
    out_images = []
    img = to_img(IMAGEURL).convert('RGB')
    # Flip the colours for one-pixel
    # "Different Image"
    for i in range(num):
        diff_img = img.copy()
        rndm_pixel_x_y = (random.randint(0, diff_img.size[0]-1), 
                          random.randint(0, diff_img.size[1]-1))
        current_color = diff_img.getpixel(rndm_pixel_x_y)
        diff_img.putpixel(rndm_pixel_x_y, current_color[::-1])
        b64img = to_base64(diff_img)
        out_images.append(json.dumps({label:'\"{0}\"'.format(b64img)}))
    return out_images

