import os
from load import *
from model import * 

checkpoint_directory = "/checkpoints"
checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")

# Create a Checkpoint that will manage two objects with trackable state,
# one we name "optimizer" and the other we name "model".
checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_directory))




status.assert_consumed()  # Optional sanity checks.
checkpoint.save(file_prefix=checkpoint_prefix)
