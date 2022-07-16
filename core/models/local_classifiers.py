import pickle
import os
import tensorflow.keras as keras
from classiFire.core.models.neural_network import NeuralNetwork
from classiFire.core.models.logreg import LogRegWrapper
from classiFire.core.models.single_assignment import SingleAssignment

_classifiers = {
	'NN': NeuralNetwork,
	'LogReg': LogRegWrapper,
	'SA': SingleAssignment
}

def load(model_path):
	with open(os.path.join(model_path, 'classifier_settings.pickle'), 'rb') as f:
		settings_dict = pickle.load(f)
		
	classifier_type = _classifiers[settings_dict['classifier_type']]
	# Need for SingleAssignment case
	if 'assignment' in settings_dict.keys():
		assignment = settings_dict['assignment']

	else:
		assignment = None

	classifier = classifier_type(
		n_input=1,
		n_output=1,
		assignment=assignment
	)
	for key in settings_dict.keys():
		if key == 'classifier_type':
			continue

		if key == 'model':
			if settings_dict['classifier_type'] == 'NN':
				classifier.model = keras.models.load_model(settings_dict[key])

			elif settings_dict['classifier_type'] == 'LogReg':
				classifier.model = pickle.load(settings_dict[key])

		else:
			classifier.__dict__[key] = settings_dict[key]

	return classifier