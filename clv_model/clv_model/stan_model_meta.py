from abc import ABCMeta
from dataclasses import dataclass
from importlib import resources

import pandas
import pystan

__all__ = ('StanModelMeta',)


class StanModelMeta(ABCMeta):
    def __init__(cls, name, bases, dct):
        cls = dataclass(cls)

        parameters = set(dct['__annotations__'])

        def _is_fitted(self) -> bool:
            return all(
                getattr(self, parameter) is not None
                for parameter in parameters
            )

        cls._is_fitted = _is_fitted

        try:
            model_name = dct['__model_name__']
        except KeyError:
            raise ValueError('__model_name__ not specified.')

        def _compile_stan_model(self) -> pystan.StanModel:
            with resources.open_text(
                'clv_model.stan_models',
                model_name + '.stan'
            ) as model_file:
                self._stan_model = pystan.StanModel(model_file)

            return self._stan_model

        cls._compile_stan_model = _compile_stan_model

        def fit(self, data: pandas.DataFrame, **kwargs) -> cls:
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
            for parameter in parameters:
                setattr(self, parameter, posteriors[parameter])

            return self

        cls.fit = fit

        def to_file(self, file_path: str) -> None:
            self._check_fit()

            pandas.DataFrame(
                data={
                    parameter: getattr(self, parameter)
                    for parameter in parameters
                }
            ).to_csv(file_path, index=False)

        cls.to_file = to_file

        @classmethod
        def from_file(cls, file_path: str) -> cls:
            parameters_df = pandas.read_csv(file_path)

            return cls(
                **{
                    parameter: parameters_df[parameter]
                    for parameter in parameters
                }
            )

        cls.from_file = from_file

        cls.__abstractmethods__ -= {'fit', '_is_fitted'}
