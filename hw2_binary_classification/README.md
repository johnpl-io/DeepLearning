## How to Run
Program:  ```python main.py```

Tests: ```python -m pytest unit_tests.py```

Output of tests: ```./output.log```

## Comments

I first made the output activation function the identity. This proved not good because allows for values that could be negative. Therefore, I decided to use a sigmoid. The hidden layers I used reLU because as explained in class, this activation function has proven to be effective. With this I was able to get to work with tuning. I adjusted the width as it does not provide a large increase in computation of power because it is used in vectorized operations. Therefore I made with the width 100. Since layers are not vectorized I only choose 3 as I throught this would be enough. I also choose 3000 iterations as increasing iterations does not hurt, but the cost function seemeed to stabilize at about 1,500 iterationss. After tuning some parameters, 
I was able to get a good output. If I used adam it probably would have converged faster. I included plots of the decision with and without L2. I also included an animated gif of training without L2 out of my own curiosity. 
