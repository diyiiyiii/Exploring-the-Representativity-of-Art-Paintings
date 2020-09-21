import tensorflow as tf
import yaml

slim = tf.contrib.slim


def _get_init_fn():
    """
    This function is copied from TF slim.

    Returns a function run by the chief worker to warm-start the training.

    Note that the init_fn is only run when initializing the model during the very
    first global step.

    Returns:
      An init function run by the supervisor.
    """
    loss_model_file="pretrained/vgg_16.ckpt"
    checkpoint_exclude_scopes="vgg_16/fc8"

    tf.logging.info('Use pretrained model %s' % loss_model_file)

    exclusions = []
    if checkpoint_exclude_scopes:
        exclusions = [scope.strip()
                      for scope in checkpoint_exclude_scopes.split(',')]

    # TODO(sguada) variables.filter_variables()
    variables_to_restore = []
    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        if not excluded:
            variables_to_restore.append(var)

    return slim.assign_from_checkpoint_fn(
        loss_model_file,
        variables_to_restore,
        ignore_missing_vars=True)


class Flag(object):
    def __init__(self, **entries):
        self.__dict__.update(entries)


def read_conf_file(conf_file):
    with open(conf_file) as f:
        FLAGS = Flag(**yaml.load(f))
    return FLAGS


if __name__ == '__main__':
    f = read_conf_file('conf/mosaic.yml')
    print(f.loss_model_file)
