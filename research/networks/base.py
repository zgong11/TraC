from functools import partial

import gymnasium as gym
import torch

import research


def reset(module):
    if hasattr(module, "reset_parameters"):
        module.reset_parameters()


class ModuleContainer(torch.nn.Module):
    CONTAINERS = []

    def __init__(self, observation_space: gym.Space, action_space: gym.Space, **kwargs) -> None:
        super().__init__()
        # save the classes and containers
        base_kwargs = {k: v for k, v in kwargs.items() if not k.endswith("_class") and not k.endswith("_kwargs")}

        output_space = observation_space
        for container in self.CONTAINERS:
            module_class = kwargs.get(container + "_class", torch.nn.Identity)
            module_class = vars(research.networks)[module_class] if isinstance(module_class, str) else module_class
            if module_class is torch.nn.Identity:
                module_kwargs = dict()
            else:
                module_kwargs = base_kwargs.copy()
                module_kwargs.update(kwargs.get(container + "_kwargs", dict()))
            # Create the module, and attach it to self
            module = module_class(output_space, action_space, **module_kwargs)
            setattr(self, container, module)

            # Set a reset function
            setattr(self, "reset_" + container, partial(self._reset, container))

            if hasattr(getattr(self, container), "output_space"):
                # update the output space
                output_space = getattr(self, container).output_space

        # Done creating all sub-modules.

    @classmethod
    def create_subset(cls, containers):
        assert all([container in cls.CONTAINERS for container in containers])
        name = "".join([container.capitalize() for container in containers]) + "Subset"
        return type(name, (ModuleContainer,), {"CONTAINERS": containers})

    def _reset(self, container: str) -> None:
        module = getattr(self, container)
        with torch.no_grad():
            module.apply(reset)

    def forward(self, x):
        # Use all of the modules in order
        for container in self.CONTAINERS:
            x = getattr(self, container)(x)
        return x


class ActorPolicy(ModuleContainer):
    CONTAINERS = ["encoder", "actor"]