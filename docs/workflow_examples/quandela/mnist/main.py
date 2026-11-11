#%%
import numpy as np
import perceval as pcvl
from perceval import BasicState, BS, GenericInterferometer, SLOSBackend, Parameter, PS, Unitary, Matrix, pdisplay
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from random import random


#%%
# 1. Load MNIST dataset
mnist = fetch_openml('mnist_784', version=1, parser="liac-arff")
X, y = mnist.data, mnist.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# 2. Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# 3. Apply PCA (e.g., retaining 95% variance or a specific number of components)
pca_component=4
pca = PCA(n_components=pca_component)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

#print(X_pca)

print(f'Original shape: {X.shape}')
print(f'Reduced shape: {X_train_pca.shape}')
print(f'Explained variance ratio sum: {pca.explained_variance_ratio_.sum():.4f}')

# 3. Scale PCA features to the range [0, 2*pi]
scaler_2pi = MinMaxScaler(feature_range=(0, 2 * np.pi))
X_train_pca_2pi = scaler_2pi.fit_transform(X_train_pca)
X_test_pca_2pi = scaler_2pi.fit_transform(X_test_pca)
isinstance(X_train_pca_2pi, np.ndarray)


#%%
def feature_maping(X):
    num_feature= len(X[0])
    num_qubit=num_feature

    #input_state = BasicState([1]*num_qubit)
    input_state = BasicState([1, 0, 1, 0])
    output_to_max = BasicState([1, 1, 0, 0])
    backend = SLOSBackend()
    
    circuit = GenericInterferometer(num_qubit,
        lambda i: BS(theta=Parameter(f"theta{i}")),
        phase_shifter_fun_gen=lambda i: PS(phi=Parameter(f"phi{i}")))
    param_circuit = circuit.get_parameters()
    params_phi_init = [random()*np.pi/2 for _ in param_circuit][4:]

    x_new =[]
    for feature in X:
        params = np.concatenate((feature,params_phi_init))
        for i, value in enumerate(params):
            param_circuit[i].set_value(value)
        backend.set_circuit(circuit)
        backend.set_input_state(input_state)
        x_new.append([backend.probability(BasicState([1, 1, 0, 0])), 
                 backend.probability(BasicState([1, 0, 1, 0])),
                 backend.probability(BasicState([1, 0, 0, 1])),
                 backend.probability(BasicState([0, 1, 1, 0]))
                 ])
    return  np.array(x_new)
         

#%%
x_tain_mapped = feature_maping(X_train_pca_2pi)
x_test_mapped = feature_maping(X_test_pca_2pi)

#%%
print("\n--- Step 4: Training the SVM Classifier ---")
# Using RBF kernel with balanced class weights
svm_clf = SVC(kernel='rbf', C=5.0, gamma='scale', random_state=42)


# Train the model on the PCA-transformed features
svm_clf.fit(x_tain_mapped, y_train)
print("Model training complete.")


print("\n--- Step 5: Model Evaluation ---")
# Generate predictions on the unseen test data
y_pred = svm_clf.predict(x_test_mapped)

# Calculate performance metrics
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred))



#%%
n=3
circuit = GenericInterferometer(n,
    lambda i: BS(theta=Parameter(f"theta{i}")),
    phase_shifter_fun_gen=lambda i: PS(phi=Parameter(f"phi{i}")))
param_circuit = circuit.get_parameters()
pdisplay(circuit)



#%%
# We create a generic interferometer (variable parameters make it universal - i.e. it can implement any unitary transformation)
n=4
circuit = GenericInterferometer(n,
    lambda i: BS(theta=Parameter(f"theta{i}")),
    phase_shifter_fun_gen=lambda i: PS(phi=Parameter(f"phi{i}")))
param_circuit = circuit.get_parameters()
params_init = [random()*np.pi/2 for _ in param_circuit]
params_init
pdisplay(circuit)
print(circuit.vars)
#%%
#print(circuit.param("phi0"))
#print(circuit.vars)
# params_init = [random()*np.pi]*(len(param_circuit)-n)
#print(len(params_init))
params_init = [random()*np.pi for _ in param_circuit]
#print(params_init)

for i, value in enumerate(params_init):
    param_circuit[i].set_value(value)

#print(circuit.param("phi0"))
#print(circuit.vars)
#%%
experiment = pcvl.Experiment(pcvl.BS()) 
experiment.with_input(pcvl.BasicState("|1,1>"))
experiment.min_detected_photons_filter(1)
computer = pcvl.SimulatedComputer("SLOS")
computer.noise = pcvl.NoiseModel(transmittance=0.05, indistinguishability=0.85)
factory = pcvl.ExecutionFactory(computer, experiment)
with computer.acquire(): 
    samples = factory.sample_count(10_000)['results']
    probs = factory.probs()['results']

print(f"Samples: {samples}")
print(f"Probabilities: {probs}")
# %%
