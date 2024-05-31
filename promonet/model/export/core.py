import inspect
import logging
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import torch

import promonet

TYPE_HASH = {bool: 0, int: 1, float: 2, str: 3, torch.Tensor: 4}


###############################################################################
# Model exporting
###############################################################################


def from_file_to_file(checkpoint, output_file):
    """Load model from checkpoint and export to torchscript"""
    model = promonet.model.Generator()
    model, *_ = torchutil.checkpoint.load(checkpoint, model)
    model.register()
    model.export_to_ts(output_file)


###############################################################################
# Exportable module base class
###############################################################################


class ExportableModule(torch.nn.Module):
    """Export module for use in Max/MSP via the IRCAM nn~ object"""

    def __init__(self) -> None:
        super().__init__()
        self._methods = []
        self._attributes = ["none"]

    def register(
        self,
        method_name: str = 'export',
        in_channels: int = promonet.INPUT_FEATURES,
        in_ratio: int = promonet.HOPSIZE,
        out_channels: int = 1,
        out_ratio: int = 1,
        input_labels: Optional[Sequence[str]] = ['features'],
        output_labels: Optional[Sequence[str]] = ['audio'],
        test_method: bool = True,
        test_buffer_size: int = 8192,
    ):
        """Register a class method as usable by nn~.

        The method must take as input and return a single 3D tensor.

        Args:
            method_name: name of the method to register
            in_channels: number of channels of the input tensor
            in_ratio: temporal compression ratio of the input tensor
            out_channels: number of channels of the output tensor
            out_ratio: temporal compression ratio of the output tensor
            input_labels: labels used by max for the inlets
            output_labels: labels used by max for the outlets
            test_method: weither the method is tested during registration or not
            test_buffer_size: duration of the test buffer
        """
        logging.info(f'Registering method "{method_name}"')
        self.register_buffer(
            f'{method_name}_params',
            torch.tensor([
                in_channels,
                in_ratio,
                out_channels,
                out_ratio,
            ]))

        if input_labels is None:
            input_labels = [
                f"(signal) model input {i}" for i in range(in_channels)
            ]
        if len(input_labels) != in_channels:
            raise ValueError(
                (f"Method {method_name}, expected "
                 f"{in_channels} input labels, got {len(input_labels)}"))
        setattr(self, f"{method_name}_input_labels", input_labels)

        if output_labels is None:
            output_labels = [
                f"(signal) model output {i}" for i in range(out_channels)
            ]
        if len(output_labels) != out_channels:
            raise ValueError(
                (f"Method {method_name}, expected "
                 f"{out_channels} output labels, got {len(output_labels)}"))
        setattr(self, f"{method_name}_output_labels", output_labels)

        if test_method:
            logging.info(f"Testing method {method_name} with nn~ API")
            x = torch.zeros(1, in_channels, test_buffer_size // in_ratio)
            y = getattr(self, method_name)(x)

            if len(y.shape) != 3:
                raise ValueError(
                    ("Output tensor must have exactly 3 dimensions, "
                     f"got {len(y.shape)}"))
            if y.shape[0] != 1:
                raise ValueError(
                    f"Expecting single batch output, got {y.shape[0]}")
            if y.shape[1] != out_channels:
                raise ValueError((
                    f"Wrong number of output channels for method \"{method_name}\", "
                    f"expected {out_channels} got {y.shape[1]}"))
            if y.shape[2] != test_buffer_size // out_ratio:
                raise ValueError(
                    (f"Wrong output length for method \"{method_name}\", "
                     f"expected {test_buffer_size//out_ratio} "
                     f"got {y.shape[2]}"))
            if y.dtype != torch.float:
                raise ValueError(f"Output tensor must be of type float, got {y.dtype}")
        else:
            logging.warn(f"Added method \"{method_name}\" without testing it.")

        self._methods.append(method_name)

    @torch.jit.export
    def get_methods(self):
        return self._methods

    @torch.jit.export
    def get_attributes(self) -> List[str]:
        return self._attributes

    def export_to_ts(self, path):
        self.eval()
        scripted = torch.jit.script(self)
        scripted.save(path)
