from typing import Optional, Tuple, Union

import torch
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.t5.configuration_t5 import T5Config
from transformers.models.t5.modeling_t5 import T5PreTrainedModel, T5Stack
from transformers.utils.model_parallel_utils import (
    assert_device_map,
    get_device_map)

import promonet


T5_1_1_CONFIG = T5Config(
    d_ff=promonet.FILTER_CHANNELS,
    d_kv=promonet.HIDDEN_CHANNELS,
    d_model=promonet.HIDDEN_CHANNELS,
    dropout_rate=promonet.P_DROPOUT,
    is_decoder=False,
    is_encoder_decoder=False,
    num_heads=promonet.N_HEADS,
    is_gated_act=True,
    dense_act_fn='gelu',
    use_cache=False
)


class T5(T5PreTrainedModel):
    authorized_missing_keys = [
        r"encoder.embed_tokens.weight",
    ]

    def __init__(self):
        super().__init__(T5_1_1_CONFIG)
        self.encoder = T5Stack(T5_1_1_CONFIG)
        self.post_init()
        self.model_parallel = False
        self.device_map = None

    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(
                len(self.encoder.block),
                range(torch.cuda.device_count()))
            if device_map is None else device_map)
        assert_device_map(self.device_map, len(self.encoder.block))
        self.encoder.parallelize(self.device_map)
        self.model_parallel = True

    def deparallelize(self):
        self.encoder.deparallelize()
        self.encoder = self.encoder.to("cpu")
        self.model_parallel = False
        self.device_map = None
        torch.cuda.empty_cache()

    def forward(
        self,
        inputs: torch.FloatTensor,
        mask: torch.FloatTensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = False
    ) -> Union[Tuple[torch.FloatTensor], BaseModelOutput]:
        return self.encoder(
            attention_mask=mask.unsqueeze(2) * mask.unsqueeze(-1),
            inputs_embeds=inputs * mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict)
