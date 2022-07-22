from PIL import Image

from resizeimage import resizeimage


with open('/Users/gouravchirkhare/Desktop/Archive/01Train_image.tif', 'r+b') as f:
    with Image.open(f) as image:
        cover = resizeimage.resize_cover(image, [4000, 4000])
        cover.save('/Users/gouravchirkhare/Desktop/Archive/02Train_image.tif', image.format)
        print("DONE")

with open('/Users/gouravchirkhare/Desktop/Archive/01Train_mask2.tif', 'r+b') as f:
    with Image.open(f) as image:
        cover = resizeimage.resize_cover(image, [4000, 4000])
        cover.save('/Users/gouravchirkhare/Desktop/Archive/02Train_mask2.tif', image.format)
        print("DONE")