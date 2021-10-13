# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" Image Utility """

import os
import numpy as np
from io import BytesIO
from PIL import Image


def imagenet_example(shape=(224, 224), data_format='channels_first',
                     path=os.path.join(os.path.dirname(__file__), 'images/example.jpg')):
    """Returns an example image and its imagenet class label.
    Parameters
    ----------
    shape : list of integers
        The shape of the returned image.
    data_format : str
        "channels_first" or "channels_last"
    path : str
        path for test image

    Returns
    -------
    image : array_like
        The example image.
    label : int
        The imagenet label associated with the image.
    """
    assert len(shape) == 2
    assert data_format in ['channels_first', 'channels_last']
    image = Image.open(path)
    image = image.resize(shape)
    image = np.asarray(image, dtype=np.float32)
    image = image[:, :, :3]
    assert image.shape == shape + (3,)
    if data_format == 'channels_first':
        image = np.transpose(image, (2, 0, 1))
    return image, 360


def save_image(image, bounds=(0, 1), data_format='channels_last'):
    """ Save image to file.
    """
    if data_format is 'channels_first':
        image = np.transpose(image, (1, 2, 0))

    if bounds == (0, 1):
        image = (image * 255).astype(np.uint8)
    from PIL import Image
    image = Image.fromarray(image)
    image.save('adversary.png')
    print('Result image saved in current directory.')


def load_mnist_image(shape=(28, 28), dtype=np.float32,
                     bounds=(0, 1), data_format='channels_last',
                     fname='mnist0.png', normalize=False):
    """Return the sample mnist image for testing
    Parameters
    ----------
    shape : list of integers
        The shape of the returned image.
    dype : np.type
        The type for loading the image
    bounds : float tuple
        the range of loaded image before normalization
    data_format : str
        "channels_first" or "channels_last"
    fname : str
        The name of sample image
    normalize : Bool
        Whether the image is needed to be normalized.
    """
    from PIL import Image

    path = os.path.join(os.path.dirname(__file__), 'images/%s' % fname)
    image = Image.open(path)
    image = np.asarray(image, dtype=dtype)
    if (data_format == 'channels_first'):
        image = image.reshape([1] + list(shape))
    else:
        image = image.reshape(list(shape) + [1])

    if bounds != (0, 255):
        image /= 255.

    return image


def load_cifar_image(shape=(32, 32), dtype=np.float32,
                     bounds=(0, 1), data_format='channels_last',
                     fname='cifar0.png', normalize=True):
    """Return the sample mnist image for testing
    Parameters
    ----------
    shape : list of integers
        The shape of the returned image.
    dype : np.type
        The type for loading the image
    bounds : float tuple
        the range of loaded image before normalization
    data_format : str
        "channels_first" or "channels_last"
    fname : str
        The name of sample image
    normalize : Bool
        Whether the image is needed to be normalized.
    """
    from PIL import Image

    path = os.path.join(os.path.dirname(__file__), 'images/%s' % fname)
    image = Image.open(path)
    image = np.asarray(image, dtype=dtype)
    if (data_format == 'channels_first'):
        image = image.reshape([3] + list(shape))
    else:
        image = image.reshape(list(shape) + [3])

    if bounds != (0, 255):
        image /= 255.

    if (normalize):
        mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        std = np.array([0.225, 0.225, 0.225]).reshape(3, 1, 1)
        image = image - mean
        image = image / std

    image = np.asarray(image, dtype=dtype)

    return image


def load_image(
        shape=(224, 224), bounds=(0, 1), dtype=np.float32,
        data_format='channels_last',
        path=os.path.join(os.path.dirname(__file__), 'images/%s' % 'example.jpg'), model_name=None):
    """Returns a resized image of target fname.
    Parameters
    ----------
    shape : list of integers
        The shape of the returned image.
    data_format : str
        "channels_first" or "channls_last".
    Returns
    -------
    image : array_like
        The example image in bounds (0, 255) or (0, 1)
        depending on bounds parameter.
    """
    assert len(shape) == 2
    assert data_format in ['channels_first', 'channels_last']
    image = Image.open(path).convert('RGB')
    image = image.resize(shape)
    image = np.asarray(image, dtype=dtype)
    image = image[:, :, :3]
    assert image.shape == shape + (3,)
    if data_format == 'channels_first':
        image = np.transpose(image, (2, 0, 1))
    preprocess_flag = True
    if model_name.startswith("paddlehub_") :
        # for paddlehub model, the colour channel should convert from RGB to BGR
        preprocess_flag = False
        image = image[..., ::-1]
    if model_name.startswith("pytorchhub_"):
        # for yolov5 from pytorchub , the image bounds should be 0, 255
        preprocess_flag = False
    if bounds != (0, 255) and image.dtype != np.uint8 and preprocess_flag:
        image /= 255.
    return image


def load_image_bytes(fname='example.jpg'):
    """ Returns a bytes stream representing the image.
    Parameters
    ----------
    fname : str
        The file name of the image.
    Returns
    -------
    image : bytes
        The example image in bytes.
    """
    import io
    with io.open(fname, 'rb') as image_file:
        image = image_file.read()
    return image


def ndarray_to_bytes(image):
    """Converting image in ndarray format to bytes."""
    if np.max(image) < 2.0:
        image = (image * 255.)
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    image_pil = Image.fromarray(image)
    bytes_output = BytesIO()
    image_pil.save(bytes_output, format='PNG')
    return bytes_output.getvalue()


def letterbox_image(
        shape=(416, 416), data_format='channels_last', fname='example.jpg'):
    """Returns a letterbox image of target fname.
    Parameters
    ----------
    shape : list of integers
        The shape of the returned image (h, w).
    data_format : str
        "channels_first" or "channls_last".
    Returns
    -------
    image : array_like
        The example image.
    """
    assert len(shape) == 2
    assert data_format in ['channels_first', 'channels_last']
    path = os.path.join(os.path.dirname(__file__), 'images/%s' % fname)
    image = Image.open(path)
    iw, ih = image.size
    h, w = shape
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', shape, (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))

    image = np.asarray(new_image, dtype=np.float32)
    image /= 255.
    image = image[:, :, :3]
    assert image.shape == shape + (3,)
    if data_format == 'channels_first':
        image = np.transpose(image, (2, 0, 1))
    return image, (h, w)


def draw_letterbox(image, prediction, original_shape=(416, 416), class_names=[], bounds=(0, 1)):
    """Draw on letterboxes on image."""
    assert len(image.shape) == 3, 'Input is a 3-dimenson numpy.ndarray'
    if bounds != (0, 1):
        import copy
        image = copy.deepcopy(image).astype(np.float32) / bounds[-1]
    if image.shape[0] == 3:
        image = np.transpose(image, [1, 2, 0])
    ih, iw = original_shape
    h, w, _ = image.shape

    scale = min(w / iw, h / ih)
    nw = int(ih * scale)
    nh = int(iw * scale)
    pad = ((w - nw) // 2, (h - nh) // 2)
    image = image[(h - nh) // 2: (h - nh) // 2 + nh,
            (w - nw) // 2: (w - nw) // 2 + nw, :]
    image = (image * 255).astype('uint8')

    image_pil = Image.fromarray(image.astype('uint8'))
    image_pil = image_pil.resize((iw, ih), Image.BICUBIC)
    new_image = np.asarray(image_pil, dtype=np.float32)
    new_image /= 255.

    if prediction == None:
        return new_image

    for idx, temp_bbox in enumerate(prediction['boxes']):
        top, left, bottom, right = temp_bbox
        top -= pad[1]
        left -= pad[0]
        bbox_re_np = np.array([top, left, bottom, right]) / scale
        bbox_rescale = bbox_re_np.astype('int').tolist()
        prediction['boxes'][idx] = bbox_rescale

    draw = draw_boxes(
        new_image, prediction['boxes'],
        prediction['classes'], prediction['scores'],
        class_names)
    return draw

def samples(dataset='imagenet', index=0, batchsize=1, shape=(224, 224),
            data_format='channels_last'):
    """Returns a batch of images and the corresponding labels.
    Parameters
    ----------
    dataset : string
        The data set to load (options: imagenet, mnist, cifar10,
        cifar100, fashionMNIST).
    index : int
        For each data set 20 example images exist. The returned batch
        contains the images with index [index, index + 1, index + 2, ...].
    batchsize : int
        Size of batch.
    shape : list of integers
        The shape of the returned image (only relevant for Imagenet).
    data_format : str
        "channels_firs" or "channels_last".
    Returns
    -------
    images : array_like
        The batch of example images.
    labels : array of int
        The labels associated with the images.
    """

    images, labels = [], []
    basepath = os.path.dirname(__file__)
    samplepath = os.path.join(basepath, 'data')
    files = os.listdir(samplepath)

    for idx in range(index, index + batchsize):
        i = idx % 20

        # get filename and label
        file = [n for n in files if '{}_{:02d}_'.format(dataset, i) in n][0]
        label = int(file.split('.')[0].split('_')[-1])

        # open file
        path = os.path.join(samplepath, file)
        image = Image.open(path)

        if dataset == 'imagenet':
            image = image.resize(shape)

        image = np.asarray(image, dtype=np.float32)

        if dataset is not 'mnist' and data_format == 'channels_first':
            image = np.transpose(image, (2, 0, 1))

        images.append(image)
        labels.append(label)

    labels = np.array(labels)
    images = np.stack(images)
    return images, labels


def onehot_like(a, index, value=1):
    """Creates an array like a, with all values
    set to 0 except one.
    Parameters
    ----------
    a : array_like
        The returned one-hot array will have the same shape
        and dtype as this array.
    index : int
        The index that should be set to `value`.
    value : single value compatible with a.dtype
        The value to set at the given index.
    Returns
    -------
    `numpy.ndarray`
        One-hot array with the given value at the given
        location and zeros everywhere else.
    """

    x = np.zeros_like(a)
    x[index] = value
    return x


def draw_boxes(image, out_boxes, out_classes, out_scores, class_names):
    """Draw output bounding boxes and scores on images."""
    from PIL import Image
    from PIL import ImageFont
    from PIL import ImageDraw
    import colorsys

    image = Image.fromarray((image * 255).astype(np.uint8))
    font_path = os.path.join(
        os.path.dirname(__file__),
        '../zoo/yolov3/model_data/FiraMono-Medium.otf')
    font = ImageFont.truetype(
        font=font_path,
        size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
    thickness = (image.size[0] + image.size[1]) // 300

    def _init_color(random_seed, num_classes):
        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(class_names), 1., 1.)
                      for x in range(len(class_names))]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                colors))
        np.random.seed(random_seed)  # Fixed seed for colors across runs.
        np.random.shuffle(colors)  # Shuffle to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.
        return colors

    colors = _init_color(10101, len(class_names))

    for i, c in reversed(list(enumerate(out_classes))):
        predicted_class = class_names[c]
        box = out_boxes[i]
        score = out_scores[i]

        label = '{} {:.2f}'.format(predicted_class, score)
        draw = ImageDraw.Draw(image)

        if draw == None:
            return image

        label_size = draw.textsize(label, font)

        top, left, bottom, right = box
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

        if top - label_size[1] >= 0:
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])

        # My kingdom for a good redistributable image drawing library.
        for i in range(thickness):
            draw.rectangle(
                [left + i, top + i, right - i, bottom - i],
                outline=colors[c])
        draw.rectangle(
            [tuple(text_origin), tuple(text_origin + label_size)],
            fill=colors[c])
        draw.text(text_origin, label, fill=(0, 0, 0), font=font)
        del draw

    return image


def draw_bounding_box_on_image(image, data_list, model_name, class_names):
    """Draw output bounding boxes, label, and scores on images for paddlehub and pytorchhub models."""
    from PIL import ImageDraw

    # For paddlehub models, the colour channel should convert from BGR to RGB.
    if model_name.startswith("paddlehub_"):
        image = image[..., ::-1]
    image = Image.fromarray((image).astype(np.uint8))
    W, H = image.size
    draw = ImageDraw.Draw(image)
    if draw == None:
        return image
    if model_name.startswith("paddlehub_"):
        for data in data_list[0]['data']:
            text = data['label'] + ":%.1f%%" % (100 * data['confidence'])
            left, right, top, bottom = data['left'], data['right'], data['top'], data['bottom']
            draw_detection_result(draw, H, W, left=left, top=top, right=right, bottom=bottom, text=text)
        return image
    elif model_name.startswith("pytorchhub_"):
        for data in data_list.pred[0]:
            if data[4] > 0.35:
                text = data_list.names[int(data[5])] + ":%.1f%%" % (100 * data[4])
                left, top, right, bottom = data[:4].cpu().numpy()
                draw_detection_result(draw, H, W, left=left, top=top, right=right, bottom=bottom, text=text, label=int(data[5]))
        return image
    elif model_name.startswith("ssd300"):
        for i, c in reversed(list(enumerate(data_list['classes']))):
            text = '{} {:.2f}'.format(class_names[c], data_list['scores'][i])
            top, left, bottom, right = data_list['boxes'][i]
            draw_detection_result(draw, H, W, left=left, top=top, right=right, bottom=bottom, text=text, label=c)
        return image
    else:
        return image

def draw_detection_result(draw, H, W, left, top, right, bottom, text, label=0):
    import colorsys
    thickness = (W + H) // 300

    def _init_color(random_seed, num_classes):
        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / 1000, 1., 1.) for x in range(1000)]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
        np.random.seed(random_seed) # Fixed seed for colors across runs.
        np.random.shuffle(colors)   # Shuffle to decorrelate adjacent classes.
        np.random.seed(None)        # Reset seed to default.
        return colors

    colors = _init_color(10101, 1000)

    label_size = draw.textsize(text=text)
    top = max(0, np.floor(top + 0.5).astype('int32'))
    left = max(0, np.floor(left + 0.5).astype('int32'))
    bottom = min(H, np.floor(bottom + 0.5).astype('int32'))
    right = min(W, np.floor(right + 0.5).astype('int32'))
    if top - label_size[1] >= 0:
        text_origin = np.array([left, top - label_size[1]])
    else:
        text_origin = np.array([left, top + 1])

    draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=colors[label])
    draw.text(xy=(text_origin[0], text_origin[1]), text=text, fill=(0, 0, 0))
    for i in range(thickness):
        draw.rectangle([left + i, top + i, right - i, bottom - i], outline=colors[label])