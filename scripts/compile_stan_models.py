from importlib import resources
import os.path
import pickle
import sys

from pystan import StanModel

sys.path.append('/app')
STAN_MODELS_PACKAGE = 'clv_model.stan_models'


def compile_stan_model(model_filename: str) -> StanModel:
    model_name = os.path.splitext(model_filename)[0]
    with resources.open_text(
        package=STAN_MODELS_PACKAGE,
        resource=model_filename
    ) as model_file:
        stan_model = StanModel(model_file, model_name=model_name)

    return stan_model


def is_compiled(model_filename: str):
    with resources.path(STAN_MODELS_PACKAGE, file_) as model_path:
        compiled = model_path.with_suffix('.pkl').is_file()
    return compiled


def pickle_model(model_filename: str, stan_model: StanModel) -> None:
    with resources.path(STAN_MODELS_PACKAGE, model_filename) as path:
        pickle_path = path.with_suffix('.pkl')
    with open(pickle_path, 'wb') as pickle_file:
        pickle.dump(stan_model, pickle_file)


if __name__ == '__main__':
    for file_ in resources.contents(STAN_MODELS_PACKAGE):
        extension = os.path.splitext(file_)[1]
        if extension != '.stan':
            continue

        if is_compiled(file_):
            continue

        stan_model = compile_stan_model(file_)
        pickle_model(file_, stan_model)
