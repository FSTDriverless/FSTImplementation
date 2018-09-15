from keras.models import load_model
import keras.backend as K
from tensorflow.python.framework import graph_io
from tensorflow.python.tools import freeze_graph
from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.training import saver as saver_lib

def convert_keras_to_pb(keras_model, out_names, models_dir, model_filename):
	model = load_model(keras_model)
	K.set_learning_phase(0)
	sess = K.get_session()
	saver = saver_lib.Saver(write_version=saver_pb2.SaverDef.V2)
	checkpoint_path = saver.save(sess, './saved_ckpt', global_step=0, latest_filename='checkpoint_state')
	graph_io.write_graph(sess.graph, '.', 'tmp.pb')
	freeze_graph.freeze_graph('./tmp.pb', '',
                          	False, checkpoint_path, out_names,
                          	"save/restore_all", "save/Const:0",
                          	models_dir+model_filename, False, "")

# the Keras model we want to convert
keras_model = './models/model.h5'

# the name of the output node in the trained network architecture
out_names = "output/Sigmoid"

# output folder and model names
models_dir = './models/'
model_filename = 'model_tf.pb'

convert_keras_to_pb(keras_model, out_names, models_dir, model_filename)