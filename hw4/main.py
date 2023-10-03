import numpy as np


class OneCycleLR:
    def __init__(
        self,
        max_lr,
        total_steps,
        pct_start=0.3,
        anneal_strategy="cos",
        div_factor=25.0,
        final_div_factor=10000.0,
    ):
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.anneal_strategy = anneal_strategy
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor
        self.step_count = 0

        self.anneal_steps = int(total_steps * pct_start)
        self.final_steps = total_steps - self.anneal_steps

    def get_lr(self):
        if self.step_count <= self.anneal_steps:
            # Warm-up phase
            if self.anneal_strategy == "linear":
                lr = self.step_count / self.anneal_steps * self.max_lr
            else:
                lr = (
                    self.max_lr
                    * (1 + np.cos(np.pi * self.step_count / self.anneal_steps))
                    / 2
                )
        else:
            # Annealing phase
            if self.anneal_strategy == "linear":
                lr = self.max_lr * (
                    1 - (self.step_count - self.anneal_steps) / self.final_steps
                )
            else:
                lr = self.max_lr * self.final_div_factor / self.div_factor
                lr *= (
                    1
                    + np.cos(
                        np.pi * (self.step_count - self.anneal_steps) / self.final_steps
                    )
                ) / 2

        self.step_count += 1
        return lr


# Example usage:
max_lr = 0.1
total_steps = 2000
scheduler = OneCycleLR(max_lr, total_steps, pct_start=0.1)

# Test the scheduler by printing the learning rates for each step
learning_rates = []
for step in range(total_steps):
    lr = scheduler.get_lr()
    learning_rates.append(lr)

import matplotlib.pyplot as plt

plt.plot(np.arange(total_steps), learning_rates)
plt.xlabel("Step")
plt.ylabel("Learning Rate")
plt.title("One Cycle Learning Rate Schedule")
plt.show()
