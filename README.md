# face-morph

## Keypoints
To label keypoints, run keypoints.py
```
python keypoints.py -i in/divi.png -j in/zap.png
```

## Morph
Json file and cropped images will be named and saved by keypoints.py, and read by morph script. To create a morph after labeling keypoints, run morph.py with cropped images.
```
python morph.py -i out/divi_crop.png -j out/zap_crop.png 
```
If you want a morph video, run 
```
python morph.py -i out/divi_crop.png -j out/zap_crop.png  -v 1
```

## Average Images
To show the average image of the danes, run mean.py
```
python mean.py
```

## Charicature
To show divi's face to the mean face, the mean face to divi's face, and the charicature image, run 
```
python scripts/divi_to_mean.py
```