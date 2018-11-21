from collections import namedtuple

# it is only to define what configurations are needed
# it contains the configuration that belongs to the network and is independent
#   of the environment it is running in
# FIXME : the decay / sample depends on examples_per_epoch too in the EnvConfig
NetworkConfig = namedtuple("SimpleNetConfig",
        ["use_gabor_wavelet", "batch_size", "epoch_per_decay", "initial_learning_rate", "learning_decay",
         "learning_momentum", "model_name", "slide_x", "slide_y", "window_width", "window_height",
         "output_dim", "max_label_num", "no_label_weight", "yes_label_weight", "leaky_alpha", "cnn_dropout_rate"])

# configuration of the running environment. The variables here are performance related
#   (yes, technically batch_size should be here but that one is encoded in the network itself
#   so it is in NetworkConfig for now
EnvConfig = namedtuple("EnvConfig",
        ["examples_per_epoch", "number_of_input_threads", "max_number_of_steps", "log_frequency", "save_frequency",
         "checkpoint_dir", "sample_dir", "min_queue_examples", "log_dir"])