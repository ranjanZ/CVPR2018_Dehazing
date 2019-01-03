# Code of the paper "Image Dehazing by Joint Estimation of Transmittance and Airlight using Bi-Directional Consistency Loss Minimized FCN"
[Project Page](http://san-santra.github.io/cvpr18w_dehaze/)

| Input        | Dehzed         | 
|:-------------:|:-------------:|
| ![input](http://san-santra.github.io/public/haze_image/ntire/41_s.jpg)| ![dehazed](http://san-santra.github.io/cvpr18w_dehaze/results/41_out_s.jpg) |

## Dependency
* For Running
    * Python 2
    * keras
    * scipy
	* numpy
	* scikit-image
	* matplotlib

## Running
```
$ cd src/
$ python main.py  <hazy_image_dir>   <output_dir>
```
This runs the code in the supplied images.
```
$python main.py  ../data/hazy_img/ ../data/out/
```

## Files
```
├── data
│   ├── hazy_img
│   │   └── lawn1_input.png
│   └── out
│       ├── ADelhi_Smog-PTI.jpg
│       ├── Delhi_Smog-PTI.jpg
│       └── TDelhi_Smog-PTI.jpg
├── models
│   └── model_weights.h5                   # Trained model
├── Readme.md
└── src
    ├── gf.py                              # guided filter
    ├── main_file.py
    ├── main.py                            # main file
    └── model.py                           # model
```

## Publication
Sanchayan Santra, Ranjan Mondal, and Bhabatosh Chanda. "Learning a Patch Quality Comparator for Single Image Dehazing." IEEE Transactions on Image Processing 27, no. 9 (2018).

BibTeX:
```
@inproceedings{mondal2018image,
  title={Image Dehazing by Joint Estimation of Transmittance and Airlight using Bi-Directional Consistency Loss Minimized FCN},
  author={Mondal, Ranjan and Santra, Sanchayan and Chanda, Bhabatosh},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops},
  pages={920--928},
  year={2018}
}
```
