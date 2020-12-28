from __future__ import annotations
from dataclasses import dataclass
from importlib import resources
from typing import Optional, Set

import pandas
import pystan

__all__ = ('StanModelBase',)


class StanModelBase:
    def __init_subclass__(cls, model_name: str, **kwargs) -> None:
        cls._stan_model: Optional[pystan.StanModel] = None
        cls.__model_name__: str = model_name
        cls = dataclass(cls)
        super().__init_subclass__(**kwargs)

    @classmethod
    def _get_parameters(cls) -> Set[str]:
        return set(cls.__annotations__)

    def _is_fitted(self) -> bool:
        return all(
            getattr(self, parameter) is not None
            for parameter in self._get_parameters()
        )

    @classmethod
    def _compile_stan_model(cls) -> pystan.StanModel:
        if cls._stan_model is not None:
            return cls._stan_model

        with resources.open_text(
            'clv_model.stan_models',
            f'{cls.__model_name__}.stan'
        ) as model_file:
            cls._stan_model = pystan.StanModel(model_file)

        return cls._stan_model

    def fit(self, data: pandas.DataFrame, **kwargs) -> StanModelBase:
        if self._is_fitted():
            return self

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
        for parameter in self._get_parameters():
            setattr(self, parameter, posteriors[parameter])

        return self

    def to_file(self, file_path: str) -> None:
        self._check_fit()

        pandas.DataFrame(
            data={
                parameter: getattr(self, parameter)
                for parameter in self._get_parameters()
            }
        ).to_csv(file_path, index=False)

    @classmethod
    def from_file(cls, file_path: str) -> StanModelBase:
        parameters_df = pandas.read_csv(file_path)

        return cls(
            **{
                parameter: parameters_df[parameter].values
                for parameter in cls._get_parameters()
            }
        )

    def posterior_mean(self) -> StanModelBase:
        self._check_fit()

        return self.__class__(
            **{
                parameter: getattr(self, parameter).mean(keepdims=True)
                for parameter in self._get_parameters()
            }
        )
