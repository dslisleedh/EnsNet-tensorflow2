# EnsNet-tensorflow2

![model](https://github.com/dslisleedh/EnsNet-tensorflow2/blob/main/model.JPG)

### 요약  

CNN의 Output과 Featuremap을 10개로 쪼개서 예측한 10개의 output, 총 11개의 output을 voting하는 ensemble CNN

### 학습방식  

1. Update CNN  

Label로 CNN의 classifier를 학습함.

2. Update subnetworks  

CNN의 variables를 동결한 후 subnetworks만 학습함
