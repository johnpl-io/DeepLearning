# Implementation of Transformer
- This project implements a transformer as in "Attention is all you need"
- In order to do this a multi_head_attention (mha) and transformer_block.
- mha is implemented with a toggable attention mask. This allows for causality 
to be preserved. 

## Testing Causal Masking
- To test the causal mask of the transformer, I put an input x 
into a mha with both a mask and not a mask enabled. After that I used 
```tape.batch_jacobian(output)``` to allow for a batched jacobian. This allows for the partial derivatives with respect to outputs to inputs to be given in a matrix form. I also used 1 element in a batch for simplicity. I found that for the masked output jacobian there was only non zero values in the first row given
using ```jacobian_with_mask[0][0]```. This represents the first token input sequences jacobian matrix. This shows the partial derivative of the output with respect to the output is zero for values that are a head of the token. This can be verified ``jacobian_with_mask[0][0]```for the  This shows that the model properly is indepedent of future elements of a sequence. Without the mask this is not true. I have tested this is unit_test.py where I extract only the zero part of the jacobian to test this. 


## Autoregressive Sample Dataset
- Now that casual masking has been verified, I constructed a synthetic dataset consisiting of the sentence "\<start> to be, or not to be, that is the question \<end>". I attempted to overfit this sentence with a causal mask (decoder only model) to show the autoregressive properties of the transformer. I choose this sentence as the vocab size is less than the input sequence, making vocab elements not having a unique next token. I believe this would be more demonstrative of the autoregressive property. I trained a 1 block model with dim_model of 512. I used a word based tokenization scheme where each element bounded by a space is given a token. I trained to model with a shifted version of the sequence at the output. I then test the model by starting with inputting \<start> into the model and appended outputs into the model until  \<end> is received. The results are in ```output.txt``` where the proper sentence is reconstructed.
## How to run

Running training: ```python main.py```

Runnning unit tests:  ```python unit_test.py```
