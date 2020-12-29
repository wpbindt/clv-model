from __future__ import annotations
from dataclasses import dataclass
from importlib import resources
import pickle
from typing import Optional, TypeVar

import numpy
import pandas
import pystan

__all__ = (
    'Parameter',
    'StanModelBase',
)

Parameter = TypeVar('Parameter')

STAN_MODELS_PACKAGE = 'clv_model.stan_models'


class StanModelBase:
    def __init_subclass__(cls, model_name: str, **kwargs) -> None:
        cls._stan_model: Optional[pystan.StanModel] = _load_stan_model(
            model_name
        )
        cls.__model_name__: str = model_name
        cls.__parameters__ = {
            name
            for name, type_ in cls.__annotations__.items()
            if type_ == Parameter
        }
        cls.__annotations__.update(
            {
                parameter: Optional[numpy.ndarray]
                for parameter in cls.__parameters__
            }
        )
        for parameter in cls.__parameters__:
            setattr(cls, parameter, None)
        cls = dataclass(cls)
        super().__init_subclass__(**kwargs)

    def is_fitted(self) -> bool:
        return all(
            getattr(self, parameter) is not None
            for parameter in self.__class__.__parameters__
        )

    @classmethod
    def _compile_stan_model(cls) -> pystan.StanModel:
        if cls._stan_model is not None:
            return cls._stan_model

        with resources.open_text(
            STAN_MODELS_PACKAGE,
            f'{cls.__model_name__}.stan'
        ) as model_file:
            cls._stan_model = pystan.StanModel(
                model_file,
                model_name=cls.__model_name__
            )

        return cls._stan_model

    def fit(self, data: pandas.DataFrame, **kwargs) -> StanModelBase:
        if self._stan_model is None:
            self._compile_stan_model()

        data_dict = {
            **dict(data),
            'N': len(data)
        }
        fit = self._stan_model.sampling(
            data=data_dict,
            **kwargs
        )

        posteriors = fit.extract(permuted=True)
        for parameter in self.__class__.__parameters__:
            setattr(self, parameter, posteriors[parameter])

        return self

    def to_file(self, file_path: str) -> None:
        self._check_fit()

        pandas.DataFrame(
            data={
                parameter: getattr(self, parameter)
                for parameter in self.__class__.__parameters__
            }
        ).to_csv(file_path, index=False)

    @classmethod
    def from_file(cls, file_path: str) -> StanModelBase:
        parameters_df = pandas.read_csv(file_path)

        return cls(
            **{
                parameter: parameters_df[parameter].values
                for parameter in cls.__parameters__
            }
        )

    def posterior_mean(self) -> StanModelBase:
        self._check_fit()

        return self.__class__(
            **{
                parameter: getattr(self, parameter).mean(keepdims=True)
                for parameter in self.__class__.__parameters__
            }
        )


def _load_stan_model(model_name: str) -> Optional[pystan.StanModel]:
    try:
        with resources.open_binary(
            STAN_MODELS_PACKAGE,
            f'{model_name}.pkl'
        ) as model_file:
            model = pickle.load(model_file)
    except FileNotFoundError:
        model = None

    return model
