import numpy as np 

def order_class_labels(class_names,targets):
	index = []
	for word in class_names:
		u = np.isin(targets,word)
		v = np.argwhere(u)
		w = v[0][0]
		index.append(w)
	index = np.argsort(index)

	class_n = []
	for x in index:
		class_n.append(class_names[x])

	return class_n


def variance_fraction(explained_variance,n_components)
#code for trying to use n_components = 0.95 variance
	retained_fraction = 0.95
	total = 0
	for i in range(0,n_components):
		total = total + explained_variance[i]
		if total > retained_fraction:
			break

	retained_components = explained_variance[:i]
	#print(total)
	#print(retained_components)
	return retained_components

