"""Train the CRBM for configuration recovery."""
import os
from collections.abc import Callable
import argparse
import tempfile
import logging
from typing import Any, Optional
import numpy as np
import h5py
import jax
import jax.numpy as jnp
from flax import nnx
from skqd_z2lgt.crbm import ConditionalRBM
from skqd_z2lgt.train_crbm import DefaultCallback, make_l2_loss_fn, cd_meanloss, train_crbm
from skqd_z2lgt.parameters import Parameters

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger('train_crbm')


def check_untrained_models(
    parameters: Parameters,
    output_filename: str
) -> list[int]:
    steps = []
    with h5py.File(output_filename, 'r+') as out:
        group = out.get('crbm') or out.create_group('crbm')
        for istep in range(parameters.skqd.n_trotter_steps):
            if f'step{istep}' not in group:
                steps.append(istep)

    return steps


def load_model(
    istep: int,
    output_filename: str
) -> tuple[ConditionalRBM, dict[str, np.ndarray]]:
    with h5py.File(output_filename, 'r') as out:
        group = out[f'crbm/step{istep}']
        model = ConditionalRBM.load(group)
        records = {key: dataset[()] for key, dataset in group['records'].items()}
    return model, records


def save_model(
    istep: int,
    model: ConditionalRBM,
    records: dict[str, np.ndarray],
    output_filename: str
):
    with h5py.File(output_filename, 'a', libver='latest') as out:
        groupname = f'crbm/step{istep}'
        try:
            del out[groupname]
        except KeyError:
            pass
        group = out.create_group(groupname)
        model.save(group)
        group = group.create_group('records')
        for key, array in records[istep].items():
            group.create_dataset(key, data=array)


def train_generator_flow(
    parameters: Parameters,
    output_filename: str,
    train_fn: Callable,
    logger: Optional[logging.Logger] = None
):
    logger = logger or logging.getLogger(__name__)

    steps_to_train = check_untrained_models(parameters, output_filename)
    if not steps_to_train:
        logger.info('All models already trained')
        return

    logger.info('Training CRBMs for Trotter steps %s', steps_to_train)

    models, records = train_fn(steps_to_train)

    for istep, model in models.items():
        save_model(istep, model, records[istep], output_filename)


def train_generator(
    parameters: Parameters,
    output_filename: str,
    logger: Optional[logging.Logger] = None
):
    conf = parameters.crbm
    model_params = {'num_h': conf.num_h, 'init_h_sparsity': conf.init_h_sparsity}
    train_params = {'l2w_weights': conf.l2w_weights, 'l2w_biases': conf.l2w_biases,
                    'batch_size': conf.batch_size, 'learning_rate': conf.learning_rate,
                    'num_epochs': conf.num_epochs, 'rtol': conf.rtol}

    def train_fn(steps_to_train):
        models = {}
        records = {}
        for istep in steps_to_train:
            with tempfile.NamedTemporaryFile() as tfile:
                tmp_out = tfile.name
            train_step_model(output_filename, istep, model_params, train_params, tmp_out)
            models[istep], records[istep] = load_model(istep, tmp_out)
            os.unlink(tmp_out)
        return models, records

    train_generator_flow(parameters, output_filename, train_fn, logger)


def train_step_model(
    filename: str,
    istep: int,
    model_params: dict[str, Any],
    train_params: dict[str, Any],
    out_filename: Optional[str] = None
):
    with h5py.File(filename, 'r', swmr=True) as source:
        dataset = source[f'data/vtx/ref_step{istep}']
        vtx_data = np.unpackbits(dataset[()], axis=-1)[..., :dataset.attrs['num_bits']]
        dataset = source[f'data/plaq/ref_step{istep}']
        plaq_data = np.unpackbits(dataset[()], axis=-1)[..., :dataset.attrs['num_bits']]
        train_u = vtx_data[:80_000]
        train_v = plaq_data[:80_000]
        test_u = vtx_data[80_000:]
        test_v = plaq_data[80_000:]

    mean_activation = np.mean(train_v, axis=0)
    mean_activation = np.where(np.isclose(mean_activation, 0.), 1.e-6, mean_activation)
    bias_init = np.log(mean_activation / (1. - mean_activation))

    num_vtx = vtx_data.shape[-1]
    num_plaq = plaq_data.shape[-1]

    rngs = nnx.Rngs(params=0, sample=1)
    model = ConditionalRBM(num_vtx, num_plaq, model_params['num_h'], rngs=rngs)
    model.bias_v.value = bias_init
    bias_h_val = np.log(model_params['init_h_sparsity'] / (1. - model_params['init_h_sparsity']))
    model.bias_h.value = jnp.full(model_params['num_h'], bias_h_val)

    model.pregenerate = True

    LOG.info('Start training model for step %d', istep)

    loss_fn = make_l2_loss_fn(cd_meanloss, train_params['l2w_weights'], train_params['l2w_biases'])
    best_model, records = train_crbm(model, train_u, train_v, test_u, test_v,
                                     train_params['batch_size'], train_params['num_epochs'],
                                     loss_fn, lr=train_params['learning_rate'],
                                     rtol=train_params['rtol'], callback=DefaultCallback())

    out_filename = out_filename or filename
    save_model(istep, best_model, records, out_filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    parser.add_argument('istep', type=int)
    parser.add_argument('--gpu')
    parser.add_argument('--out-filename')
    parser.add_argument('--num-h', type=int, default=256)
    parser.add_argument('--l2w-weights', type=float, default=1.)
    parser.add_argument('--l2w-biases', type=float, default=0.2)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--init-h-sparsity', type=float, default=0.01)
    parser.add_argument('--num-epochs', type=int, default=10)
    parser.add_argument('--rtol', type=float)
    options = parser.parse_args()

    if options.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = options.gpu
    jax.config.update('jax_enable_x64', True)

    mparams = {'num_h': options.num_h, 'init_h_sparsity': options.init_h_sparsity}
    tparams = {'l2w_weights': options.l2w_weights, 'l2w_biases': options.l2w_biases,
               'batch_size': options.batch_size, 'learning_rate': options.learning_rate,
               'num_epochs': options.num_epochs, 'rtol': options.rtol}
    train_step_model(options.filename, options.istep, mparams, tparams,
                     out_filename=options.out_filename)
