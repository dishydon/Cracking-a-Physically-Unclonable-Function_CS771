import numpy as np
from sklearn.svm import LinearSVC as SVC
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import accuracy_score

# You are allowed to import any submodules of sklearn as well e.g. sklearn.svm etc
# You are not allowed to use other libraries such as scipy, keras, tensorflow etc

# SUBMIT YOUR CODE AS A SINGLE PYTHON (.PY) FILE INSIDE A ZIP ARCHIVE
# THE NAME OF THE PYTHON FILE MUST BE submit.py
# DO NOT INCLUDE OTHER PACKAGES LIKE SCIPY, KERAS ETC IN YOUR CODE
# THE USE OF ANY MACHINE LEARNING LIBRARIES OTHER THAN SKLEARN WILL RESULT IN A STRAIGHT ZERO

# DO NOT CHANGE THE NAME OF THE METHODS my_fit, my_predict etc BELOW
# THESE WILL BE INVOKED BY THE EVALUATION SCRIPT. CHANGING THESE NAMES WILL CAUSE EVALUATION FAILURE

# You may define any new functions, variables, classes here
# For example, functions to calculate next coordinate or step length

################################
# Non Editable Region Starting #
################################
def my_fit( Z_train ):
################################
#  Non Editable Region Ending  #
################################

	# Use this method to train your model using training CRPs
	# The first 64 columns contain the config bits
	# The next 4 columns contain the select bits for the first mux
	# The next 4 columns contain the select bits for the second mux
	# The first 64 + 4 + 4 = 72 columns constitute the challenge
	# The last column contains the response
	def get_combo(Z):
		return np.concatenate([Z[:,0].reshape(-1,1) * Z[:,1:4], Z[:,1].reshape(-1,1) * Z[:,2:4], Z[:,2].reshape(-1,1) * 
                               Z[:,3:4] , Z[:,:], (Z[:,0]*Z[:,1]*Z[:,2]).reshape(-1,1), (Z[:,0]*Z[:,1]*Z[:,3]).reshape(-1,1),
                               (Z[:,0]*Z[:,2]*Z[:,3]).reshape(-1,1),(Z[:,1]*Z[:,2]*Z[:,3]).reshape(-1,1),(Z[:,0]*Z[:,1]*Z[:,2]*Z[:,3])
                               .reshape(-1,1)], axis=1)

	def new_features(Z):
		Z1 = np.einsum('ij,ik->ikj', get_combo(Z[:,64:68]) - get_combo(Z[:,68:72]), Z[:,0:64])
		Z1 = Z1.reshape(Z1.shape[0],-1)
		Z2 = get_combo(Z[:,64:68]) - get_combo(Z[:,68:72])
		return np.concatenate([Z1, Z2], axis=1)
    
	model = SVC(penalty='l2', C=100, max_iter=int(1e3), tol=1e-4, loss = 'squared_hinge', dual = False)
	features = new_features(Z_train[:,:72])
	model.fit(features , Z_train[:,-1])
	
	return model					# Return the trained model


################################
# Non Editable Region Starting #
################################
def my_predict( X_tst, model ):
################################
#  Non Editable Region Ending  #
################################

	# Use this method to make predictions on test challenges
	def get_combo(Z):
		return np.concatenate([Z[:,0].reshape(-1,1) * Z[:,1:4], Z[:,1].reshape(-1,1) * Z[:,2:4], Z[:,2].reshape(-1,1) * 
                               Z[:,3:4] , Z[:,:], (Z[:,0]*Z[:,1]*Z[:,2]).reshape(-1,1), (Z[:,0]*Z[:,1]*Z[:,3]).reshape(-1,1),
                               (Z[:,0]*Z[:,2]*Z[:,3]).reshape(-1,1),(Z[:,1]*Z[:,2]*Z[:,3]).reshape(-1,1),(Z[:,0]*Z[:,1]*Z[:,2]*Z[:,3])
                               .reshape(-1,1)], axis=1)

	def new_features(Z):
		Z1 = np.einsum('ij,ik->ikj', get_combo(Z[:,64:68]) - get_combo(Z[:,68:72]), Z[:,0:64])
		Z1 = Z1.reshape(Z1.shape[0],-1)
		Z2 = get_combo(Z[:,64:68]) - get_combo(Z[:,68:72])
		return np.concatenate([Z1, Z2], axis=1)
  	
	pred = model.predict(new_features(X_tst[:,:72]))
	return pred


# X_train = np.loadtxt("secret_train.dat")
# X_test = np.loadtxt("secret_test.dat")

# model = my_fit(X_train)
# print(accuracy_score(my_predict(X_test, model), X_test[:,-1]))