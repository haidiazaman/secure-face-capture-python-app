 TRAIN_TRANSFORMS_FACE = [
        # transforms after crop (if crop == True)
        Compose([
            Resize(
                size=INPUT_SIZE,
                antialias=RESIZE_ANTIALIAS
            ),
            RandomHorizontalFlip(),
        ]),
        # pre-transforms before crop (if crop == True), else this will not be used
        Compose([
            RandomApply([
                BoundingBoxAffine(
                    width_range=(0.9,1.1),
                    height_range=(0.9,1.1),
                )],
                p=1.0
            ),
            RandomApply([
                ColorJitter(
                    brightness=0.15,
                    contrast=0.15,
                    saturation=0.15,
                )],
                p=0.66
            ),
            RandomApply([
                ImageCompression(
                    quality_lower=75,
                    quality_upper=95,
                )],
                p=0.66
            ),
            RandomApply([
                RandomRotation(
                    degrees=20,
                    fill=127
                )],
                p=1.0
            ),
        ]),
    ]
    VAL_TRANSFORMS_FACE = [
        Compose([
            BoundingBoxAffine(
                    width_range=(0.92,1.08),
                    height_range=(0.92,1.08),
                ),
            Resize(
                size=INPUT_SIZE,
                antialias=RESIZE_ANTIALIAS
            ),
        ])
    ]


    TRAIN_TRANSFORMS_FULL = [
        # transforms after crop (if crop == True)
        Compose([
            RandomResizedCrop(
                size=INPUT_SIZE,
                scale=(0.8,1.0),
                ratio=(0.8,1.2),
                antialias=RESIZE_ANTIALIAS
            ),
            RandomApply([
                ColorJitter(
                    brightness=0.15,
                    contrast=0.15,
                    saturation=0.15,
                )],
                p=0.66
            ),
            RandomApply([
                ImageCompression(
                    quality_lower=75,
                    quality_upper=95,
                )],
                p=0.66
            ),
            RandomApply([
                RandomRotation(
                    degrees=20,
                    fill=127
                )],
                p=1.0
            ),
            RandomHorizontalFlip(),
        ]),
        # pre-transforms before crop (if crop == True), else this will not be used
        Identity(),
    ]
    VAL_TRANSFORMS_FULL = [
        Compose([
            Resize(
                size=INPUT_SIZE,
                antialias=RESIZE_ANTIALIAS
            ),
        ])
    ]
    
    
import cv2
import numpy as np
import torch
import torchvision

class ImageCompression():
    # ImageCompression augmentation from albumentations adapted to torchvision
    def __init__(
        self,
        quality_lower=75,
        quality_upper=100,
        compression_type='jpeg',
    ):
        self.quality_lower = quality_lower
        self.quality_upper = quality_upper

    def __call__(self, data):
        if isinstance(data, tuple):
            image = data[0]
            image = np.transpose(np.array(image),(1,2,0))
            if not image.ndim == 2 and image.shape[-1] not in (1, 3, 4):
                raise TypeError("ImageCompression transformation expects 1, 3 or 4 channel images.")
            return tuple([self.image_compression(image, np.random.randint(self.quality_lower, self.quality_upper), cv2.IMWRITE_JPEG_QUALITY), *data[1:]])
        else:
            image = data
            image = np.transpose(np.array(image),(1,2,0))
            if not image.ndim == 2 and image.shape[-1] not in (1, 3, 4):
                raise TypeError("ImageCompression transformation expects 1, 3 or 4 channel images.")
            return self.image_compression(image, np.random.randint(self.quality_lower, self.quality_upper), cv2.IMWRITE_JPEG_QUALITY)
        
    
    def image_compression(self, img, quality, quality_flag):
        input_dtype = img.dtype
        assert input_dtype == np.uint8, "Only support 'uint8' data type"
        _, encoded_img = cv2.imencode(".jpg", img, (int(quality_flag), quality))
        img = cv2.imdecode(encoded_img, cv2.IMREAD_UNCHANGED)
        return torch.from_numpy(np.transpose(img,(2,0,1)))

    
class BoundingBoxAffine():
    def __init__(self, width_range=(0.8, 1.2), height_range=(0.8, 1.2)):
        self.width_factor_range = width_range
        self.height_factor_range = height_range

    def __call__(self, data):
        
        if not isinstance(data, tuple):
            raise TypeError("BoundingBoxAffine transformation expects a bbox input. Only an image input was given")
        
        image_height, image_width = data[0].shape[1:]
        bbox = data[1]
                
        if not isinstance(bbox, torchvision.datapoints.BoundingBox):
            raise TypeError("BoundingBoxAffine transformation expects a bbox of type 'torchvision.datapoints.BoundingBox'")

        # Randomly sample a scaling factor
        width_factor = np.random.uniform(self.width_factor_range[0], self.width_factor_range[1])
        height_factor = np.random.uniform(self.height_factor_range[0], self.height_factor_range[1])

        # Calculate new bounding box coordinates
        left, top, right, bottom = bbox
        width = right - left
        height = bottom - top

        new_width = int(width * width_factor)
        new_height = int(height * height_factor)

        # Ensure the bounding box remains inside the image bounds
        left = max(0, int(left - (new_width - width) / 2))
        right = min(image_width, int(right + (new_width - width) / 2))
        top = max(0, int(top - (new_height - height) / 2))
        bottom = min(image_height, int(bottom + (new_height - height) / 2))
        
        bbox = torchvision.datapoints.BoundingBox([left, top, right, bottom], format="XYXY", spatial_size=(image_height, image_width))
        
        if len(data) > 2:
            return tuple([data[0], bbox, *data[2:]])
        else:
            return (data[0], bbox)

    
class Identity():
    # Identity augmentation that returns input without any augmentation (used to fulfill function structure)
    def __init__(self):
        pass
    
    def __call__(self, data):
        return data
    
# How to use:
# bbox = torchvision.datapoints.BoundingBox([LEFT,TOP,RIGHT,BOTTOM], format="XYXY", spatial_size=[IMG_WIDTH,IMG_HEIGHT])
# image, bbox = TRANSFORMS(image, bbox)
# [LEFT,TOP,RIGHT,BOTTOM] - List of integers defining the bbox coordinates
# [IMG_WIDTH, IMG_HEIGHT] - List of integers defining the full image dimensions
# TRANSFORMS - A torchvision Compose object, defining the transform pipeline