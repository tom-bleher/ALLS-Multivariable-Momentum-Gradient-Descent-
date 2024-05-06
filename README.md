### Project description
This version of the optimization algorithm utilizes momentum gradient descent such that instead of the usual gradient descent: 

$$W_{n+1}=W_{n}-\gamma \nabla f(W_{n})$$

By adding a 'momentum' term, we arrive at the modified expression:

$$W_{n+1}=W_{n}+\beta(W_{n}-W_{n-1})-\gamma \nabla f(W_{n})$$

Where $\gamma$ is the learning rate as previously seen in the classic gradient descent algorithm, and $\beta$ is the new momentum constant. 

In Python this mathematical formula takes on the following form:
```python
self.new_focus = self.focus_history[-1] + (self.momentum*(self.focus_history[-1]-(self.focus_history[-2])) - self.focus_learning_rate*self.count_focus_der[-1]
```

This upgrade promises to help the optimization:
- Take less step until optimization
- Avoid getting stuck in local maximum points

### Faster optimization
As evident by comparing the graphs for the optimization process, the momentum acceleration plot (right) reaches the maximum faster in fewer steps.
| ![Momentum Gradient Descent](Media/momentum2.png) | | ![Vanilla Gradient Descent](Media/vanilla2.png) |
|:--------------------------------------------------:|---|:-----------------------------------------------:|
|            Momentum Gradient Descent             |   |             Vanilla Gradient Descent            |


Unlike classic gradient descent, momentum gradient descent takes less sharp turns. Essentially, where gradient descent depends on the previous gradient, momentum gradient descent incorporates a moving average of past gradients, allowing it to smooth out variations in the optimization.

<br>
<div align="center">
<img src="Media/vanilla_gd.jpg" width="50%" height="50%" />
</div>

### Absolute optimization
In order to test the promise of momentum gradient descent to find the absolute maximum and minimum points, I tested using the following function:
$$C(f,\phi_{2})=(0.1(f+\phi_{2}))^{2}\cdot \sin(0.01(f+\phi_{2}))$$
When plotting it we see it takes on the form:
<br>
<div align="center">
<img src="Media/Pasted image 20240123181900.png" width="50%" height="50%" />
</div>

Focusing on the region near zero, I get the following optimization test region with local and 'absolute' minimum points.
<br>
<div align="center">
<img src="Media/Pasted image 20240124012635.png" width="50%" height="50%" />
</div>

I will initialize the algorithm at the red point as seen above. Where a classic gradient descent will get stuck in the local minimum, the momentum gradient descent will be able to optimize to the greater 'absolute' minimum surpassing the local minimum and saddle point.

Now, running the algorithm as seen in `momentum_main.py` I will verify my assumptions. 

Using vanilla gradient descent ($\beta=0$), we arrive at the local minimum point as expected:
<br>
<div align="center">
<img src="Media/Pasted image 20240124013223.png" width="50%" height="50%" />
</div>

Using momentum gradient we arrive at the greater minimum point:
<br>
<div align="center">
<img src="Media/Pasted image 20240124013253.png" width="50%" height="50%" />
</div>

As shown, the algorithm was able to get to the global minimum. Mission complete? Not exactly. The system is sensitive to the learning rates and momentum constants and will not always arrive at the optimal solution.
