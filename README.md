# FastDataAcquisition-ObjDetSeg
Code for our paper "Fast Training Data Acquisition for Object Detection and Segmentation using Black Screen Luminance Keying"  

The first script takes videos of the objects in front of a black screen with high light absorption and saves the RGBA images, the alpha representing the object mask. The second script creates a dataset consisting of randomly placed objects on randomly selected background images. It uses alpha blending and strong augmentation. The objects are rotated and scaled randomly.  

Our YCB-V LUMA dataset, recorded with a black screen, is available here https://huggingface.co/datasets/tpoellabauer/YCB-V-LUMA. If you find our work useful, please consider citing our paper.  

## Sample Usage
```
python3 gen_stylegan_train_set.py --source ./black_screen_videos --add_alpha --scale=0.8 --dest ./black_screen_images --resolution=512x512  

python3 gen_chroma_set_with_bgreplacement.py --crops ./black_screen_images --bgs bg_images --dest ./dataset --resolution=512x512
```