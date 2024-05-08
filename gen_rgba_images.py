import os
import zipfile
import click
import ffmpegio
import PIL.Image
import numpy as np
import cv2 as cv
import io
import sys
import re
import json
from glob import glob
from typing import Callable, Optional, Tuple, Union
from pathlib import Path
from enum import Enum

def error(msg):
    print('Error: ' + msg)
    sys.exit(1)
    
#----------------------------------------------------------------------------  

def parse_tuple(s: str) -> Tuple[int, int]:
    '''Parse a 'M,N' or 'MxN' integer tuple.

    Example:
        '4x2' returns (4,2)
        '0,1' returns (0,1)
    '''
    m = re.match(r'^(\d+)[x,](\d+)$', s)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    raise ValueError(f'cannot parse tuple {s}')
    
#----------------------------------------------------------------------------    
    
def maybe_min(a: int, b: Optional[int]) -> int:
    if b is not None:
        return min(a, b)
    return a

#----------------------------------------------------------------------------

def file_ext(name: Union[str, Path]) -> str:
    return str(name).split('.')[-1]

#----------------------------------------------------------------------------

def is_image_ext(fname: Union[str, Path]) -> bool:
    ext = file_ext(fname).lower()
    return f'.{ext}' in PIL.Image.EXTENSION # type: ignore
    
#----------------------------------------------------------------------------

def open_image_zip(source, *, max_images: Optional[int]):
    with zipfile.ZipFile(source, mode='r') as z:
        input_images = [str(f) for f in sorted(z.namelist()) if is_image_ext(f)]

        # Load labels.
        labels = {}
        if 'dataset.json' in z.namelist():
            with z.open('dataset.json', 'r') as file:
                labels = json.load(file)['labels']
                if labels is not None:
                    labels = { x[0]: x[1] for x in labels }
                else:
                    labels = {}

    max_idx = maybe_min(len(input_images), max_images)

    def iterate_images():
        with zipfile.ZipFile(source, mode='r') as z:
            for idx, fname in enumerate(input_images):
                with z.open(fname, 'r') as file:
                    img = PIL.Image.open(file) # type: ignore
                    img = np.array(img)
                yield dict(img=img, label=labels.get(fname))
                if idx >= max_idx-1:
                    break
    return max_idx, iterate_images()


#----------------------------------------------------------------------------
    
def open_dest(dest: str) -> Tuple[str, Callable[[str, Union[bytes, str]], None], Callable[[], None]]:
    dest_ext = file_ext(dest)

    if dest_ext == 'zip':
        if os.path.dirname(dest) != '':
            os.makedirs(os.path.dirname(dest), exist_ok=True)
        zf = zipfile.ZipFile(file=dest, mode='w', compression=zipfile.ZIP_STORED)
        def zip_write_bytes(fname: str, data: Union[bytes, str]):
            zf.writestr(fname, data)
        return '', zip_write_bytes, zf.close
    else:
        # If the output folder already exists, check that is is
        # empty.
        #
        # Note: creating the output directory is not strictly
        # necessary as folder_write_bytes() also mkdirs, but it's better
        # to give an error message earlier in case the dest folder
        # somehow cannot be created.
        if os.path.isdir(dest) and len(os.listdir(dest)) != 0:
            error('--dest folder must be empty')
        os.makedirs(dest, exist_ok=True)

        def folder_write_bytes(fname: str, data: Union[bytes, str]):
            os.makedirs(os.path.dirname(fname), exist_ok=True)
            with open(fname, 'wb') as fout:
                if isinstance(data, str):
                    data = data.encode('utf8')
                fout.write(data)
        return dest, folder_write_bytes, lambda: None

#----------------------------------------------------------------------------

# crop image and resize to specified resolution
def transform_image(image, resolution, add_alpha=True, object_scale=1.0, debug=False):
    w_res = resolution[0]
    h_res = resolution[1]
    # shape: [1, 2160, 3840, 3] -> [2160, 3840, 3]
    img = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    mask = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    cropped_img = None
    
    ret, thresh = cv.threshold(mask, 30,255,cv.THRESH_BINARY)
    # add alpha channel
    if add_alpha:
        img = cv.cvtColor(img, cv.COLOR_RGB2RGBA)
        img[:, :, 3] = thresh
        
    contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    if len(contours) != 0:
        ##----------------------------scaling-------------------------------
        contour = max(contours, key=cv.contourArea)
        # BB
        bb_x,bb_y,bb_w,bb_h = cv.boundingRect(contour)
        
        # get the offset of the boundingbox size and the specified resolution
        w_scale = (w_res*object_scale)/bb_w # penalize the ratio, so that BB fit at least 90 % 
        h_scale = (h_res*object_scale)/bb_h # of resolution size

        scale = min(w_scale, h_scale)
        
        # image width and height
        height = img.shape[0]
        width = img.shape[1] 
                                
        # resize image
        img = cv.resize(img, (int(scale * width), int(scale * height)), interpolation=cv.INTER_LANCZOS4)  
           
        ##----------------------------center cropping-------------------------------
        # image width and height
        height = img.shape[0]
        width = img.shape[1] 
        
        # get boundingbox
        contours, hierarchy = cv.findContours(img[:,:,-1], cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        contour = max(contours, key=cv.contourArea)
        bb_x,bb_y,bb_w,bb_h = cv.boundingRect(contour)  
        
        # get center of the boundingbox
        center_x = int((bb_x*2+bb_w)/2)
        center_y = int((bb_y*2+bb_h)/2)
        #print("before c_x, c_y, c_w, c_h: ", center_x, center_y, center_w, center_h)
        
        
        # adjust coordinates and width, height according to the resolution and image size
        w_shift = round(w_res / 2)
        h_shift = round(h_res / 2)
        
        # update coordinates    
        begin_x = center_x - w_shift
        end_x = center_x + w_shift
        begin_y = center_y - h_shift
        end_y = center_y + h_shift
        
        # add 1 additionally pixel, if resolution is odd    
        if w_res % 2 != 0:
            end_x += 1     
        if h_res % 2 != 0:
            end_y += 1    
        
        left_pad = 0
        right_pad = 0
        top_pad = 0
        bottom_pad = 0
        # adjust if BB outside the image (left)
        if begin_x < 0:
            left_pad = np.abs(begin_x)
            begin_x = 0                

        # adjust if BB outside the image (up)
        if begin_y < 0:
            top_pad = np.abs(begin_y)
            begin_y = 0
            
        # adjust if BB outside the image (right)
        x_offset = end_x - width
        if x_offset > 0:
            right_pad = x_offset
            end_x = width
                  
        # adjust if BB outside the image (down)
        y_offset = end_y - height
        if y_offset > 0:
            bottom_pad = y_offset
            end_y = height 
            
        
        if debug:  
            img_a = cv.rectangle(img_a, (begin_x,begin_y), (end_x, end_y), (255, 0,0), 1)
            img_a = cv.circle(img_a, (center_x, center_y), 1, (0,0,255), -1)
            cv.imwrite("./img_debug.png", img_a) 
                  
        # crop image with BB
        cropped_img = img[begin_y:end_y, begin_x:end_x]
        # pad cropped image
        cropped_img = cv.copyMakeBorder(cropped_img, top_pad, bottom_pad, left_pad, right_pad, cv.BORDER_CONSTANT, None, [0,0,0]) 
    return cropped_img


#----------------------------------------------------------------------------

class YCBV(Enum):
    coffee = 1
    cracker = 2
    sugar = 3
    tomato = 4
    mustard = 5
    tuna = 6
    chocolate = 7
    jello_strawberry = 8
    potted_meat = 9
    banana = 10
    can = 11
    bleach_cleanser = 12
    bowl = 13
    mug = 14
    driller = 15
    wood_block = 16
    scissor = 17
    pen = 18    
    small_clamp = 19
    big_clamp = 20
    foam_brick = 21
    
class Glasses(Enum):
    glass1 = 1
    glass2 = 2
    glass3 = 3
    glass4 = 4
    glass5 = 5
    glass6 = 6
    glass7 = 7
    glass8 = 8
    glass9 = 9
    glass10 = 10
    glass11 = 11
    glass12 = 12
    glass13 = 13
    glass14 = 14
    glass15 = 15
    
#----------------------------------------------------------------------------

# get bounding box on rgba image
def get_BB(image):
    mask = image[:, :, -1]
    _, thresh = cv.threshold(mask, 150,255,cv.THRESH_BINARY)
    contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    
    contour = max(contours, key=cv.contourArea)
    # BB
    bb_x,bb_y,bb_w,bb_h = cv.boundingRect(contour)
    x1 = np.max((0, bb_x))
    y1 = np.max((0, bb_y))
    x2 = np.max((0, bb_w))
    y2 = np.max((0, bb_h))
    area = cv.contourArea(contour)
    return [int(x1), int(y1), int(x2), int(y2)], int(area)
    
#----------------------------------------------------------------------------


@click.command()
@click.pass_context
@click.option('--source', help='Directory or archive name for input dataset', required=True, metavar='PATH')
@click.option('--add_alpha', is_flag=True, help='Option for adding the mask as alpha channel', required=True, metavar=bool)
@click.option('--scale', help='scale the object', metavar=float)
@click.option('--dest', help='Output directory or archive name for output dataset', required=True, metavar='PATH')
@click.option('--resolution', help='Output resolution (e.g., \'512x512\')', metavar='WxH', type=parse_tuple)
def convert_dataset(
    ctx: click.Context,
    source: str,
    add_alpha: bool,
    scale,
    dest: str,
    resolution: str
):

    PIL.Image.init() # type: ignore
    
    # configuration
    debug=False
    
    max_images = 1000 # max images per object
    
    if scale is not None:
        scale = float(scale)

    if dest == '':
        ctx.fail('--dest output filename or directory must not be an empty string')
    # training destination
    dest_train = dest.replace(".zip", "%dx%d.zip" % (resolution[0], resolution[1]))
    archive_root_dir_train, save_bytes_train, close_dest_train = open_dest(dest_train)    

    try:
        source_split = source.split(',')
    except:
        source_split = [source]

    if resolution is None: resolution=(1024, 1024)
    
    annotation_train = {}
    # add categories
    i = 1
    categories_train = []
    while i < 22:
        categories_train.append({'id': i, 'name': i, 'supercategory': archive_root_dir_train})
        i += 1
    annotation_train['categories'] = categories_train
    
    images_train = []
    annotations_train = []
    
    backgrounds = glob(f"./backgrounds/backgrounds/*.png")
    
    dataset_attrs = None
    labels_train = []
    print("Start processing images")
    train_idx = 0
    # iterate over videos in folder
    for root, dirs, files in os.walk(source):
        files = [f for f in files if f.lower().endswith('.mov') or f.lower().endswith('.mp4')]
        for f in files:
            images_per_obj = 0
            # get label
            obj_name = f.split('.')[0].lower()
            if "glass" in obj_name:
                label = Glasses[obj_name].value
            else:
                label = YCBV[obj_name].value
            print('Obj class: ', label)
            print("File: ", f)
            
            cap = cv.VideoCapture(os.path.join(root, f))
            # Read until video is completed
            while(cap.isOpened()):
              # Capture frame-by-frame
              ret, image = cap.read()
              if ret == True:
                # do we have enough samples of the object?
                if images_per_obj >= max_images:
                    break
                                            
                # crop frames and resize to specified resolution
                if scale is not None:
                    img = transform_image(image, resolution, add_alpha, object_scale=scale)
                else :
                    img = transform_image(image, resolution, add_alpha)
                                                          

                # Transform may drop images.
                if img is None:
                    continue   
                
                # add background
                bg_idx = np.random.randint(0, len(backgrounds)-1, 1)
                #print(bg_idx)
                bg = cv.imread(backgrounds[bg_idx[0]])
                bg = cv.resize(bg, resolution, interpolation=cv.INTER_LANCZOS4)
                bg = cv.cvtColor(bg, cv.COLOR_BGR2RGB)
                bg_rgba = cv.cvtColor(bg, cv.COLOR_RGB2RGBA)
                bg_rgba[:, :, 3] = np.zeros_like(bg_rgba[:, :, 3])
                mask = np.repeat(np.expand_dims(img[:,:,-1], axis=2), 4, axis=2) 
                img = np.where(mask > 100, img, bg_rgba)  
                
                if debug:    
                    img_tmp = img[:,:,:-1]
                    mask = np.where(img[:,:,-1] > 100, 1.0, 0.0)
                    mask_tmp = np.repeat(np.expand_dims(mask,2),3, axis=2)
                    img_res = mask_tmp * img_tmp 
                    print("img_tmp: ", img_tmp.shape)
                    img_res = np.float32(img_res)
                    img_res = cv.cvtColor(img_res, cv.COLOR_BGR2RGB)
                    cv.imwrite("./img_res_blending.png", img_res)   
                    cv.imwrite("./img.png", img)#cv.cvtColor(img, cv.COLOR_RGBA2RGB))
                    cv.imwrite("./mask.png", mask)
                    exit()    
                                  

                # Error check to require uniform image attributes across
                # the whole dataset.
                cur_image_attrs = {
                    'width': img.shape[1],
                    'height': img.shape[0],
                    'channels': img.shape[2]
       	        }
                if dataset_attrs is None:
                    dataset_attrs = cur_image_attrs
                    width = dataset_attrs['width']
                    height = dataset_attrs['height']
                    if width != height:
                        error(f'Image dimensions after scale and crop are required to be square.  Got {width}x{height}')
                    if dataset_attrs['channels'] not in [3, 4]:
                        error('Input images must be stored as RGB or RGBA')
                    if width != 2 ** int(np.floor(np.log2(width))):
                        error('Image width/height after scale and crop are required to be power-of-two')
                elif dataset_attrs != cur_image_attrs:
                    err = [f'  dataset {k}/cur image {k}: {dataset_attrs[k]}/{cur_image_attrs[k]}' for k in dataset_attrs.keys()]

                bbox, area = get_BB(img)
                height, width = int(img.shape[0]), int(img.shape[1])
                # save image bytes
                img = PIL.Image.fromarray(img, { 3: 'RGB' , 4: 'RGBA'}[img.shape[2]])
                image_bits = io.BytesIO()
                img.save(image_bits, format='png', compress_level=0, optimize=False)
                
                train_idx += 1 
                idx_str = f'{train_idx:08d}'
                archive_fname = f'{idx_str[:5]}/img{idx_str}.png'
                # Save the image as an uncompressed PNG.
                save_bytes_train(os.path.join(archive_root_dir_train, archive_fname), image_bits.getbuffer())
                # gen coco format
                images_train.append({'id': train_idx, 'file_name': archive_fname, 'width': width, 'height': height, 'date_captured': '', 'license': 1, 'coco_url': '', 'flickr_url': ''})
                annotations_train.append({'id': train_idx, 'image_id': train_idx, 'category_id': label, 'iscrowd': 0, 'area': area, 'bbox': bbox, 'segmentation': [], 'width': width, 'height': height, 'ignore': 'false'})
                labels_train.append([archive_fname, label])
                # update image per object counter
                images_per_obj += 1 # images sampled from this object in total
                                    
         
              # Break the loop
              else: 
                break
             
            # When everything done, release the video capture object
            cap.release()
             
            # Closes all the frames
            cv.destroyAllWindows()
            print(images_per_obj) 

    annotation_train['images'] = images_train
    annotation_train['annotations'] = annotations_train
    metadata_train = {
        'labels': labels_train if all(x is not None for x in labels_train) else None
    }
    
    save_bytes_train(os.path.join(archive_root_dir_train, 'scene_gt_coco.json'), json.dumps(annotation_train))
    save_bytes_train(os.path.join(archive_root_dir_train, 'dataset.json'), json.dumps(metadata_train))
    close_dest_train()
    
if __name__ == "__main__":
    convert_dataset()
	
