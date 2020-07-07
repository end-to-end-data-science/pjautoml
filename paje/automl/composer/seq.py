# -*- coding: utf-8 -*-
""" Composer module

This module implements and describes the 'Pipeline' composer.  This composer
sequentially operates on 'Elements' component or 'Composers'.

For more information about the Composer concept see [1].

.. _paje_arch Paje Architecture:
    TODO: put the link here
"""

from paje.automl.composer.composer import Composer


class Seq(Composer):
    # def build_impl(self, **config):
    #     """
    #     The only parameter is config with the dic of each component.
    #     :param config
    #     :return:
    #     """
    #     configs = [{} for _ in self.components]  # Default value
    #
    #     if 'configs' in self.config:
    #         configs = self.config['configs']
    #     self.components = self.components.copy()
    #     zipped = zip(range(0, len(self.components)), configs)
    #     for idx, compo_config in zipped:
    #         # TODO: setar showwarns?
    #         newconfig = compo_config.copy()
    #         newconfig['random_state'] = self.random_state
    #         self.components[idx] = self.components[idx].build(**newconfig)

    @staticmethod
    def sampling_function(config_spaces):
        return [x.sample() for x in config_spaces]

