from paje.base.component import Component
from paje.ml.element.element import Element


class Resampler(Element):
    def apply_impl(self, data):
        # TODO: generalize this to resample all fields (xyzuvwpq...)
        X, y = self.model.fit_resample(*data.Xy)
        return data.updated(self, X=X, y=y)

    def use_impl(self, data):
        return data
