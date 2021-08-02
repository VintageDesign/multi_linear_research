import sys
sys.path.insert(0, r'F:\repos\multi_linear_research\jackson\Libraries\pylds')
from pylds.models import DefaultLDS
import numpy.random as npr
import matplotlib.pyplot as plt
import numpy as np

D_obs = 1       # Observed data dimension
D_latent = 2	# Latent state dimension
D_input = 0	    # Exogenous input dimension
T = 2000  	    # Number of time steps to simulate

true_model = DefaultLDS(D_obs, D_latent, D_input)
inputs = npr.randn(T, D_input)
data, stateseq = true_model.generate(T, inputs=inputs)

plt.plot(data)
plt.show()

train = data[:1800]
test = data[1800:]

print(len(train), len(test))

# Create a separate model and add the observed data
test_model = DefaultLDS(D_obs, D_latent, D_input)
test_model.add_data(train)

N = 20
for i in range(N):
    test_model.resample_model()

gen, states = test_model.generate(2000)
gen = gen[1800:]

plt.figure()
plt.plot(test, color="blue", lw=2, label="observed")
plt.plot(gen, color="orange", label="generated")
plt.xlim(0,200)
plt.show()

plt.plot(np.abs((test-gen)/gen), label="observed")
plt.show()