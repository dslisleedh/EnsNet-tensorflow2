# EnsNet-tensorflow2

![model](https://github.com/dslisleedh/EnsNet-tensorflow2/blob/main/model.JPG)

## NOTE !!! 
Reworked at 20221123  
Current Training state is SO UNSTABLE.  
I'll fix it later. Please add issues or PR if you find any problems.  
Thank you.  

Update(20221123 - 17:50): I found that the validation loss converged faster when the batch size was small. Maybe Random{augmentation_layer} layer applys same augmentation rate/direction to batch? 

## How to run

You can run train code by `train.py`  


    conda env create -f environment.yaml
    conda activate ensnet
    python .\train.py dataset=mnist # [mnist, fashion_mnist, cifar10] Or you can use --multirun. Check hydra
    
