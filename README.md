### Project description
This version of the optimization algorithm utilizes momentum gradient descent such that instead of the usual gradient descent: 

$$W_{n+1}=W_{n}-\gamma \nabla f(W_{n})$$

By adding a 'momentum' term, we arrive at the modified expression:

$$W_{n+1}=W_{n}+\beta(W_{n}-W_{n-1})-\gamma \nabla f(W_{n})$$

Where $\gamma$ is the learning rate as previously seen in the classic gradient descent algorithm, and $\beta$ is the new momentum constant. 

In python code this mathematical formula takes on the following form:
```python
self.new_focus = self.focus_history[-1] + (self.momentum*(self.focus_history[-1]-(self.focus_history[-2])) - self.focus_learning_rate*self.count_focus_der[-1]
```

This upgrade promises to help the optimization:
- Take less step until optimization
- Avoid getting stuck in local maximum points

### Faster optimization
<br>
<div align="center">
<img src="Media/vanilla_gd.jpg" width="50%" height="50%" />
</div>
Unlike classic gradient descent, momentum gradient descent takes less sharp turns. Essentially, where gradient descent depend on the previous gradient, momentum gradient descent incorporates a moving average of past gradients, allowing it to smooth out variations in the optimization.

![[Media/vanilla_gd.jpg|center|400]]

### Absolute optimization
In order to test the promise of momentum gradient descent to find the absolute maximum and minimum points, I tested using the following function:
$$C(f,\phi_{2})=(0.1(f+\phi_{2}))^{2}\cdot \sin(0.01(f+\phi_{2}))$$
When plotting it we see it takes on the form:
<br>
<div align="center">
<img src="Media/Pasted image 20240123181900.png" width="50%" height="50%" />
</div>

Focusing on the region near zero, I get the following optimization test region with a local and 'absolute' minimum points.
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

As shown, the algorithm was able to get to the global minimum. Mission complete? Not exactly. The system is sensitive to the learning rates and momentum constants and will not always arrive to the optimal solution.

### Additional code updates

My changes include:
- Breaking up the repetitive code to functions 
- Implementing more `numpy` alternatives for faster computation

The code now utilizes `numpy` arrays as an alternative for python lists for faster computation.

```python
        self.third_dispersion_der_history = np.array([])
        self.second_dispersion_der_history = np.array([])
        self.focus_der_history = np.array([])
        self.total_gradient_history = np.array([])
```


For better organization in the function `process_images()`, I broke up the code into two functions.

```python
    def write_value(self):
        self.new_focus = int(round(np.clip(self.new_focus, self.FOCUS_LOWER_BOUND, self.FOCUS_UPPER_BOUND)))
        self.new_second_dispersion = int(round(np.clip(self.new_second_dispersion, self.SECOND_DISPERSION_LOWER_BOUND, self.SECOND_DISPERSION_UPPER_BOUND)))
        self.new_third_dispersion = int(round(np.clip(self.new_third_dispersion, self.THIRD_DISPERSION_LOWER_BOUND, self.THIRD_DISPERSION_UPPER_BOUND)))
        self.focus_history = np.append(self.focus_history, [self.new_focus])
        self.second_dispersion_history = np.append(self.second_dispersion_history, [self.new_second_dispersion])
        self.third_dispersion_history = np.append(self.third_dispersion_history, [self.new_third_dispersion])

        mirror_values[0] = self.new_focus
        dispersion_values[0] = self.new_second_dispersion
        dispersion_values[1] = self.new_third_dispersion

        with open(MIRROR_FILE_PATH, 'w') as file:
            file.write(' '.join(map(str, mirror_values)))
        with open(DISPERSION_FILE_PATH, 'w') as file:
            file.write(f'order2 = {dispersion_values[0]}\n')
            file.write(f'order3 = {dispersion_values[1]}\n')
        self.upload_files() # send files to second computer
        QtCore.QCoreApplication.processEvents()
```

```python
    def plot_reset(self):
        self.plot_curve.setData(self.iteration_data, self.count_history)
        self.focus_curve.setData(self.der_iteration_data, self.focus_der_history)
        self.second_dispersion_curve.setData(self.der_iteration_data, self.second_dispersion_der_history)
        self.third_dispersion_curve.setData(self.der_iteration_data, self.third_dispersion_der_history)
        self.total_gradient_curve.setData(self.der_iteration_data, self.total_gradient_history)

        self.n_images_count_sum = 0
        self.mean_count_per_n_images  = 0
        img_mean_count = 0
```

Using these, the `process_images()` function takes on the following form:

```python
def process_images(self, new_images):

self.initialize_image_files()
new_images = [image_path for image_path in new_images if os.path.exists(image_path)]
new_images.sort(key=os.path.getctime)

for image_path in new_images:
	img_mean_count = self.calc_xray_count(image_path)
	self.n_images_count_sum += np.sum(img_mean_count)

	self.run_count += 1

	if self.run_count % self.n_images == 0:
		self.mean_count_per_n_images = np.mean(img_mean_count)
		self.count_history = np.append(self.count_history, self.mean_count_per_n_images)
		self.n_images_run_count += 1
		self.iteration_data = np.append(self.iteration_data, self.n_images_run_count)

		if self.n_images_run_count == 1 or self.n_images_run_count ==2:
			print('-------------')  

			if self.run_count == 1:                        
				self.focus_history = np.append(self.focus_history, self.initial_focus)      
				self.second_dispersion_history = np.append(self.second_dispersion_history, self.initial_second_dispersion)
				self.third_dispersion_history = np.append(self.third_dispersion_history, self.initial_third_dispersion)
				print(f"initial values are: focus {self.focus_history[-1]}, second_dispersion {self.second_dispersion_history[-1]}, third_dispersion {self.third_dispersion_history[-1]}")
				print(f"initial directions are: focus {self.random_direction[0]}, second_dispersion {self.random_direction[1]}, third_dispersion {self.random_direction[2]}")
				self.initial_optimize()
				
			if self.run_count == 2:                        
				self.initial_optimize()
				
		if self.n_images_run_count > 2:
			self.n_images_dir_run_count += 1
			self.optimize_count()
		self.write_values() # write values and send via FTP connection
		print(f"mean_count_per_{self.n_images}_images {self.count_history[-1]}, current values are: focus {self.focus_history[-1]}, second_dispersion {self.second_dispersion_history[-1]}, third_dispersion {self.third_dispersion_history[-1]}")
		self.plot_reset() # update plotting lists and reset variables
		print('-------------')             
```
