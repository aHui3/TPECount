# TPECount
Text-Guided Patch-Level Exemplar Selection for Zero-Shot Counting

## Dataset
TPECount is trained and tested on the **FSC-147 dataset**. The dataset is available for download from [here](https://github.com/cvlab-stonybrook/LearningToCountEverything). The dataset is organized as follows:
```
TPECount/datasets/FSC147_384_V2/
├── images_384_VarV2/  
├── gt_density_map_adaptive_384_VarV2/        
├── annotation_FSC147_384_V2.json
├── ImageClasses_FSC147.txt   
└── Train_Test_Val_FSC_147.json
```

## Pretrained Models
The pre-trained weights that enable reproduction of the results reported in the paper are available for download [here](https://drive.google.com/file/d/1Jb66xPfngi3g3fJIiIP4JUicPKSwPXTm/view?usp=drive_link). Please place the downloaded weights in the `TPECount/checkpoints/` directory.

## Testing
To test the model, run the following command in the `TPECount` directory:
```
python test.py
```


