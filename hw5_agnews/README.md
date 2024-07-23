
# Reporting

## Performance
| Model | Validation Accuracy | Test Accuracy   |
|----------|---------------------|-----------------|
| all-mpnet-base-v2 with MLP head (keras head) |  91.65%              | 92.34%          |
| all-mpnet-base-v2 with MLP head (non-keras head) |     90.92%         | 90.78%          |
## Notes
I used a pre-trained model from sentence transformers called all-mpnet-base-v2
and then fed this into an MLP head (one with keras, and one with my custom MLP) to compare. I did both just to see if there was a difference between my custom and the keras version. They are both the same architecture, interestingly, the keras one performs slightly better than the non-keras one. However, keras uses 'glorot initialization' instead of 'he initialization' as default for their dense layers. I am not sure if this played a major role in the difference between the two. Also, the keras MLP trained much faster for a similar number of epochs. This may be do to some optimizations keras does under the hood that is not applied to my custom MLP. I added weight decay and dropout as I noticed overfitting can happen quickly for this model. Both models are ran for about 10 epochs. I believe that I should look into the keras source code more to see if there is a more direct reason why it has slightly better performance than my custom MLP head.

# How to run

```python ag_news.py```

- I also put in an html of a notebook file I used so it is possible to see outputs as I ran the program.

