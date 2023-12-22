# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Streaming images and labels from datasets created with dataset_tool.py."""

import os
import numpy as np
import zipfile
import PIL.Image
import json
import torch
import dnnlib
import pandas
import math
#import imageio
import cv2

import training.training_loop
from training.utils import circular_mask

try:
    import pyspng
except ImportError:
    pyspng = None

import sky_image_generator
import secondary_channels
import training.utils

#----------------------------------------------------------------------------

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
        name,                   # Name of the dataset.
        raw_shape,              # Shape of the raw image data (NCHW).
        max_size    = None,     # Artificially limit the size of the dataset. None = no limit. Applied before xflip.
        use_labels  = False,    # Enable conditioning labels? False = label dimension is zero.
        xflip       = False,    # Artificially double the size of the dataset via x-flips. Applied after max_size.
        random_seed = 0,        # Random seed to use when applying max_size.
    ):
        self._name = name
        self._raw_shape = list(raw_shape)
        self._use_labels = use_labels
        self._raw_labels = None
        self._label_shape = None

        # Apply max_size.
        self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)
        if (max_size is not None) and (self._raw_idx.size > max_size):
            np.random.RandomState(random_seed).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:max_size])

        # Apply xflip.
        self._xflip = np.zeros(self._raw_idx.size, dtype=np.uint8)
        if xflip:
            self._raw_idx = np.tile(self._raw_idx, 2)
            self._xflip = np.concatenate([self._xflip, np.ones_like(self._xflip)])

        self._mask = circular_mask(raw_shape[1:])

    def _get_raw_labels(self):
        if self._raw_labels is None:
            self._raw_labels = self._load_raw_labels() if self._use_labels else None
            if self._raw_labels is None:
                self._raw_labels = np.zeros([self._raw_shape[0], 0], dtype=np.float32)
            assert isinstance(self._raw_labels, np.ndarray)
            assert self._raw_labels.shape[0] == self._raw_shape[0]
            assert self._raw_labels.dtype in [np.float32, np.int64]
            if self._raw_labels.dtype == np.int64:
                assert self._raw_labels.ndim == 1
                assert np.all(self._raw_labels >= 0)
        return self._raw_labels

    def close(self): # to be overridden by subclass
        pass

    def _load_raw_image(self, raw_idx): # to be overridden by subclass
        raise NotImplementedError

    def _load_raw_labels(self): # to be overridden by subclass
        raise NotImplementedError

    def __getstate__(self):
        return dict(self.__dict__, _raw_labels=None)

    def __del__(self):
        try:
            self.close()
        except:
            pass

    def __len__(self):
        return self._raw_idx.size

    def __getitem__(self, idx): # outputs a masked image in the range [0, 1] (mostly - at least in the LDR case; HDR crosses both limits)
        image = self._load_raw_image(self._raw_idx[idx])
        assert isinstance(image, np.ndarray)
        assert list(image.shape) == self.image_shape
        dtype = image.dtype
        assert dtype in [np.uint8, np.float32]

        image *= self._mask

        if dtype == np.uint8:
            image = image.astype(np.float32) / 255 # [0, 1]
        elif dtype == np.float32:
            #image = training.training_loop.unstretch(training.utils.log_transform(image)) # to roughly [0, 1+]
            pass
        else:
            raise Exception('Unexpected image.dtype ' + str(image.dtype))

        if self._xflip[idx]:
            assert image.ndim == 3 # CHW
            image = image[:, :, ::-1]

        return image.copy(), self.get_label(idx)

    def get_label(self, idx):
        label = self._get_raw_labels()[self._raw_idx[idx]]
        if label.dtype == np.int64:
            onehot = np.zeros(self.label_shape, dtype=np.float32)
            onehot[label] = 1
            label = onehot
        return label.copy()

    def get_details(self, idx):
        d = dnnlib.EasyDict()
        d.raw_idx = int(self._raw_idx[idx])
        d.xflip = (int(self._xflip[idx]) != 0)
        d.raw_label = self._get_raw_labels()[d.raw_idx].copy()
        return d

    @property
    def name(self):
        return self._name

    @property
    def image_shape(self):
        return list(self._raw_shape[1:])

    @property
    def num_channels(self):
        assert len(self.image_shape) == 3 # CHW
        return self.image_shape[0]

    @property
    def resolution(self):
        assert len(self.image_shape) == 3 # CHW
        assert self.image_shape[1] == self.image_shape[2]
        return self.image_shape[1]

    @property
    def label_shape(self):
        if self._label_shape is None:
            raw_labels = self._get_raw_labels()
            if raw_labels.dtype == np.int64:
                self._label_shape = [int(np.max(raw_labels)) + 1]
            else:
                self._label_shape = raw_labels.shape[1:]
        return list(self._label_shape)

    @property
    def label_dim(self):
        assert len(self.label_shape) == 1
        return self.label_shape[0]

    @property
    def has_labels(self):
        return any(x != 0 for x in self.label_shape)

    @property
    def has_onehot_labels(self):
        return self._get_raw_labels().dtype == np.int64

#----------------------------------------------------------------------------

class ImageFolderDataset(Dataset):
    def __init__(self,
        path,                   # Path to directory or zip or csv.
        resolution      = None, # Ensure specific resolution, None = highest available, setting a resolution forces resizing (one-time preprocessing)
        normalize_azimuth=False,# Rotate the image so that the sun azimuth is 180 deg (one-time preprocessing) 
        **super_kwargs,         # Additional arguments for the Dataset base class.
    ):
        self._path = path
        self._resolution = resolution
        self._normalize_azimuth = normalize_azimuth
        self._zipfile = None

        if os.path.isdir(self._path):
            self._type = 'dir'
            self._all_fnames = {os.path.relpath(os.path.join(root, fname), start=self._path) for root, _dirs, files in os.walk(self._path) for fname in files}
        elif self._file_ext(self._path) == '.zip':
            self._type = 'zip'
            self._all_fnames = set(self._get_zipfile().namelist())
        elif self._file_ext(self._path) == '.csv':
            self._type = 'csv'
            csv = pandas.read_csv(self._path)
            csv = csv.sort_values(by=['img_fname']) # TODO maybe sort by time
            
            if csv['img_fname'][0][0] == '/':
                self._all_fnames = csv['img_fname'].to_list()
            else:
                # not absolute path - prepend the dirname of the csv
                dirname = os.path.dirname(self._path)
                self._all_fnames = csv['img_fname'].map(lambda rel: os.path.join(dirname, rel))
            assert os.path.isfile(self._all_fnames[0])

            # Get the exposure fix multipliers (extra columns in the .csv)
            #   EV_shift          ... (negative! and +12?) EV steps that best match the HDR images to the clear sky model with the same Sun position (fitted per image in fit_exposure.ipynb)
            #   multiplier_to_ldr ... a multiplier roughly match the LDR .jpg image generated by PTGui (multiplier found per-image in fit_exposure.ipynb)
            #   manual_EV_shift   ... manually selected per-shoot (NOT per-image) exposures for roughly equal brightness (a dict in fit_exposure.ipynb)
            if 'manual_EV_shift' in csv.columns:
                print('reading manual_EV_shift from csv')
                self._all_multipliers = np.array(csv['manual_EV_shift'].to_list()) 
            else:
                # newer auto_processed dataset has consistent exposures, no per-shoot correction necessary
                print('using zero manual_EV_shift')
                self._all_multipliers = pandas.Series(np.tile([0], csv.shape[0]))

            self._all_multipliers = 2**(self._all_multipliers) # if EV steps, convert to a multiplier

            self._all_azimuths = csv['sun_azimuth'].to_numpy()
            self._all_elevations = csv['sun_elevation'].to_numpy()

            # replace HDR filenames with corresponding JPGs for now (debug)
            #self._all_fnames = [file.replace('EXR', 'JPG').replace('_hdr','').replace('.exr','.jpg') for file in self._all_fnames]
        else:
            raise IOError('Path must point to a directory or zip')

        PIL.Image.init()
        self._image_fnames = [fname for fname in self._all_fnames if self._file_ext(fname) in {**PIL.Image.EXTENSION, **{'.exr': 'EXR'}}]
        if len(self._image_fnames) == 0:
            raise IOError('No image files found in the specified path')

        name = os.path.splitext(os.path.basename(self._path))[0]
        raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0).shape)
        if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
            raise IOError('Image files do not match the specified resolution')
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(self._path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        if self._type == 'csv':
            return open(fname, 'rb')
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)
    
    @staticmethod
    def rotate_image(image, angle):
        # https://stackoverflow.com/a/9042907
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        return result
    
    def _load_raw_image(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        multiplier = self._all_multipliers[raw_idx]
        wanted_size = self._resolution

        extension = fname.split('.')[-1]

        if self._normalize_azimuth:
            rotated_fname = fname.replace(
                '.' + extension,
                '.rot_az180' + '.' + extension
            )

            if not os.path.isfile(rotated_fname):
                print('rotating', fname, 'to sun_azimuth == 180 as', rotated_fname)
                ori_sun_azimuth = self._all_azimuths[raw_idx]
                assert extension == 'exr' # TODO png support?
                cv2.imwrite(rotated_fname+'.tmp.'+extension, self.rotate_image(cv2.imread(fname, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH), 180 - ori_sun_azimuth))
                os.rename(rotated_fname+'.tmp.'+extension, rotated_fname)

            fname = rotated_fname

        if wanted_size is not None:
            resized_fname = fname.replace(
                '.' + extension,
                '.resized' + str(wanted_size) + '.' + extension
            )
            
            if not os.path.isfile(resized_fname):
                print('resizing', fname, 'to', wanted_size, 'as', resized_fname)
                
                if extension == 'exr':
                    #imageio.imwrite(resized_fname, cv2.resize(imageio.imread(fname), (wanted_size, wanted_size)))
                    #imageio.imwrite(resized_fname.replace('.exr', '.png'), (imageio.imread(fname)[:,:,:3]*255).clip(0, 255).astype(np.uint8))
                    cv2.imwrite(resized_fname+'.tmp.'+extension, cv2.resize(cv2.imread(fname, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH), (wanted_size, wanted_size)))
                    os.rename(resized_fname+'.tmp.'+extension, resized_fname)
                else:
                    # TODO: this resizing is possibly incorrect - it does not work in linear colour space - we should convert it to linear, then resize, then back to the original space (assuming sRGB). But maybe it wouldn't make a noticable difference as the original was already LDR (discretised to 256 values)
                    PIL.Image.open(fname)\
                    .resize((wanted_size, wanted_size))\
                    .save(resized_fname)

            fname = resized_fname

        with self._open_file(fname) as f:
            if pyspng is not None and self._file_ext(fname) == '.png':
                image = pyspng.load(f.read())
            elif self._file_ext(fname) == '.exr':
                image = cv2.imread(fname, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[:,:,::-1] # BGR -> RGB
                
                if image.shape[2] == 4:
                    image = image[:,:,:3] # remove alpha channel if present
                
                image *= multiplier
                image = training.training_loop.unstretch(training.utils.log_transform(image)) # to roughly [0, 1+]
                #image = np.clip(image, 0, 1) # DEBUG

                #print('image', image.min(), image.mean(), np.median(image), image.max())
            else:
                image = np.array(PIL.Image.open(f))
        #print('image.shape', image.shape)
        assert image.shape[0] == image.shape[1] # square
        if wanted_size is not None:
            assert image.shape[0] == wanted_size

        if image.ndim == 2:
            image = image[:, :, np.newaxis] # HW => HWC
        image = image.transpose(2, 0, 1) # HWC => CHW
        return image

    def _load_raw_labels(self):
        fname = 'dataset.json'
        if fname not in self._all_fnames:
            return None
        with self._open_file(fname) as f:
            labels = json.load(f)['labels']
        if labels is None:
            return None
        labels = dict(labels)
        labels = [labels[fname.replace('\\', '/')] for fname in self._image_fnames]
        labels = np.array(labels)
        labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
        return labels

#----------------------------------------------------------------------------


import diskcache
ClearSkyDataset_mapping = diskcache.Cache(
    directory=os.path.join(os.getenv('CACHE_DIR'), 'diskcache_ClearSkyDataset'),
    size_limit=100 * 2**30, # 100 GB
)
@ClearSkyDataset_mapping.memoize()
def generate_clear_sky_image(resolution, azimuth, elevation):
    # fixed parameters, optimized/fitted on "2019-08-10_1000_santa_cruz_villa_nuova/1K_EXR/IMG_2000_hdr.exr" using the "sky_image_generator_py.ipynb" notebook
    exposure, visibility, ground_albedo = -9.17947373e+00,  1.00012133e+02,  5.95418177e-03
    
    model_img = sky_image_generator.generate_image(
        resolution,
        elevation / 180 * np.pi, # elevation
        math.fmod(360 + 270 - azimuth, 360) / 180 * np.pi, # azimuth
        visibility, # visibility (in km)
        ground_albedo # ground albedo
    )
    
    exposure += 12 # exposure fix

    img = model_img * np.power(2, exposure)
    #print('img', img.min(), img.mean(), img.max())
    return img # [0, 1?]
    
def generate_clear_sky_image_and_secondary_channels(resolution, secondary_channels, azimuth, elevation):
    img = generate_clear_sky_image(resolution, azimuth, elevation) / 255
    
    img = img[..., ::-1] # RGB -> BGR

    # add secondary/guiding channels
    polar_distance = (secondary_channels.polar_distance(secondary_channels.phi, secondary_channels.theta, math.fmod(360 + 270 - azimuth, 360) / 180 * np.pi, elevation / 180 * np.pi) + 1) / 2 # ([-1, 1] to [0, 1?])

    img_with_secondary_channels = np.concatenate([
        img, # RGB
        np.expand_dims(polar_distance, 2),
    ], axis=2)
    
    image = img_with_secondary_channels.transpose(2, 0, 1) # HWC => CHW

    return image


class ClearSkyDataset(Dataset):
    def __init__(self,
        path,                   # Path to directory or zip or csv.
        resolution      = None, # Ensure specific resolution, None = highest available.
        normalize_azimuth = False, # set azimuth to 180 deg
        **super_kwargs,         # Additional arguments for the Dataset base class.
    ):
        self._path = path
        self._resolution = resolution

        if self._file_ext(self._path) == '.csv':
            self._type = 'csv'
            csv = pandas.read_csv(self._path)
            self.csv = csv.sort_values(by=['img_fname']) # TODO maybe sort by time
            self._all_fnames = self.csv['img_fname'].to_list()
            # replace HDR filenames with corresponding JPGs for now
            self._all_fnames = [file.replace('EXR', 'JPG').replace('_hdr','').replace('.exr','.jpg') for file in self._all_fnames]

            azimuths = self.csv['sun_azimuth'].to_numpy()
            if normalize_azimuth:
                azimuths *= 0
                azimuths += 180
            
            azimuths = np.fmod(azimuths, 360)
            
            elevations = self.csv['sun_elevation'].to_numpy()
            
            self._all_azimuths = azimuths
            self._all_elevations = elevations
        else:
            raise IOError('Path must point to a directory or zip')

        PIL.Image.init()
        #import ipdb; ipdb.set_trace()
        filtered_indices = [idx for idx, fname in enumerate(self._all_fnames) if self._file_ext(fname) in PIL.Image.EXTENSION]

        self._image_fnames = [fname for idx, fname in enumerate(self._all_fnames) if idx in filtered_indices]
        if len(self._image_fnames) == 0:
            raise IOError('No image files found in the specified path')
        
        self._image_azimuths = [azimuth for idx, azimuth in enumerate(self._all_azimuths) if idx in filtered_indices]
        self._image_elevations = [elevation for idx, elevation in enumerate(self._all_elevations) if idx in filtered_indices]

        self.secondary_channels = secondary_channels.SecondaryChannels(self._resolution)

        name = os.path.splitext(os.path.basename(self._path))[0]+'_clear'
        raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0).shape)

        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _load_raw_image(self, raw_idx):
        azimuth = self._image_azimuths[raw_idx]
        elevation = self._image_elevations[raw_idx]
        return generate_clear_sky_image_and_secondary_channels(self._resolution, self.secondary_channels, azimuth, elevation)
    
    def _load_raw_labels(self):
        return None
