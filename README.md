# U-Net for Semantic Segmentation on Unbalanced Aerial Imagery
Fork of https://github.com/amirhosseinh77/UNet-AerialSegmentation
Read the article at [Towards data science](https://towardsdatascience.com/u-net-for-semantic-segmentation-on-unbalanced-aerial-imagery-3474fa1d3e56)
[Kaggle Dataset](https://www.kaggle.com/humansintheloop/semantic-segmentation-of-aerial-imagery)

## Training 
```
!python train.py --num_epochs 2 --batch 2 --loss focalloss
```

## Docker
On Windows
```
docker build -t image-predictor .
winpty docker run -v "${PWD}"/images:/app/images -v "${PWD}"/predictions:/app/predictions image-predictor python predict.py images/IMAGENAME.jpg
```

## Data Classes
- Building: #3C1098 (Dark Purple)
- Land (unpaved area): #8429F6 (Light Purple)
- Road: #6EC1E4 (Light Blue)
- Vegetation: #FEDD3A (Yellow)
- Water: #E2A929 (Orange)
- Unlabeled: #9B9B9B (Gray)