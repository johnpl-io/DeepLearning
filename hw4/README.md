# DeepLearning
## Validation and Test accuracy    

| Dataset  | Validation Accuracy | Test Accuracy   | Parameters
|----------|---------------------|-----------------|-----------------|
| CIFAR-10 | 82.03%              | 79.13%          |1,920,778|
| CIFAR-100| 62.89%              | 64.44%          |2,262,628|


## SOTA

My SOTA was originally around 90%. I believed this was possible after doing research about the state of the art. In the [Stanford](https://dawn.cs.stanford.edu/benchmark/) dawn AI competetion where they were able to achieve 94%. A source the winner used was able to achieve above 90% in 20 epochs. Since I used a batch size of 512 I believed that it could be possible in theoretically around 2k iterations. There design used a 9 layer resnet commonly referred to as resnet9. It is inspired by the paper [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) in which they use larger resnets. The design has an interesting structure as seen [here](https://github.com/davidcpage/cifar10-fast), I tried to implement it along the way but found that it was difficult to encorporate well enough with my classifier class and did not have much improvement. For my current architecture for CIFAR-10, I use a similar network that has two prep layers, three conv layers, and when flattened layer. I attempt to avoid convolving shortcut connections if possible. For CIFAR 100 I have a slightly bigger structure with the same principle. 

## input data normalization
I divided all of my pixels by 255 and made all images have a mean of 0 and stdev of 1. This technique is supposed allow for my easy learning. Interestingly I noticed increased contrast when this was applied which may be a reason for this. 

## Weight Decay

I used weight decay in both, but in CIFAR 100 I increased it. I found my model overfitting, so I tried to use it to improve accuracy of val set. For cifar 10 overfitting was not as much of an issue. 

## Design Choices and Experiments
Similar to other resnet9 implementations, I introduced maxpooling and a final average pooling layer. This allows for my model to be much less memory intensive as the pooling options I choose allowed for decrease in filter size. I also use a drop layer right before the output. From my research, dropout typically should be used in the fully connected layers but sources seemed to not be fully certain about this. I did not change my CIFAR 10 model do to restrictions in memory, and knowing that a model of this size is workable. Therefore I decided to tune learning rate, increasing it to allow for larger range of movement toward a minimum. Having a high learning rate in the beginning is typically of many fast learning networks. After this I got a bit stuck hovering around 60-70% accuracy. Eventually I was able to get into the mid to high 70s and touch the 80s when applying data augmentation each iteration. This proves to be slow, but allows for better generalization. Unfortunately, I ran out of time to do so on CIFAR 100 so I just implemented data augmentation in the beginning. I performed the data augmentation in the resnet paper, pad 4 pixels on each size flip randomly and crop. For CIFAR 100 I was having problems with overfitting. I tried to increase my network about due to increased number of features. I also tried to increase the dropout of the last layer but this did not help much. I lastly tried to increase weight decay to help avoid overfitting. I unfortatnely was not able to get out of this having at one time a training acc that was 90% top k but with only 60-70 test acc. To get better accuracy I believe encorporating an lr finder and lr scheduler could have helped as this is what many of the models in the competition have done. 

## How to run
- cifar 10 python cifar10.py
- cifar 100 python cifar100.py
- unit tests python -m pytest unit_test/*.py



