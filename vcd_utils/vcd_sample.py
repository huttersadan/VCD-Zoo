import copy
import inspect
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch import nn

from transformers.generation.logits_process import (
    LogitsProcessorList,
)
from transformers.modeling_outputs import (BaseModelOutputWithPast,
                                           CausalLMOutputWithPast,
                                           SequenceClassifierOutputWithPast)
from torch.nn import CrossEntropyLoss
from transformers.generation.stopping_criteria import (
    StoppingCriteria,
    StoppingCriteriaList,
    validate_stopping_criteria,
)
import transformers
from transformers.generation.utils import SampleOutput, SampleEncoderDecoderOutput, SampleDecoderOnlyOutput

from transformers.generation.streamers import BaseStreamer
from transformers.generation.utils import GenerateNonBeamOutput
from transformers.generation.utils import GenerateEncoderDecoderOutput
from transformers.generation.utils import GenerateDecoderOnlyOutput
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from transformers.models.llava.modeling_llava import LLAVA_INPUTS_DOCSTRING,LlavaCausalLMOutputWithPast,_CONFIG_FOR_DOC
from transformers import LlamaForCausalLM
@add_start_docstrings_to_model_forward(LLAVA_INPUTS_DOCSTRING)
@replace_return_docstrings(output_type=LlavaCausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
def forward(
    self,
    input_ids: torch.LongTensor = None,
    pixel_values: torch.FloatTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    vision_feature_layer: Optional[int] = None,
    vision_feature_select_strategy: Optional[str] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    mask_idx: Optional[torch.Tensor] = None,
    masking_scheme=None,
    input_ids_length = 10,
) -> Union[Tuple, LlavaCausalLMOutputWithPast]:
    r"""
    Args:
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

    Returns:

    Example:

    ```python
    >>> from PIL import Image
    >>> import requests
    >>> from transformers import AutoProcessor, LlavaForConditionalGeneration

    >>> model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")
    >>> processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

    >>> prompt = "<image>\nUSER: What's the content of the image?\nASSISTANT:"
    >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
    >>> image = Image.open(requests.get(url, stream=True).raw)

    >>> inputs = processor(text=prompt, images=image, return_tensors="pt")

    >>> # Generate
    >>> generate_ids = model.generate(**inputs, max_length=30)
    >>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    "\nUSER: What's the content of the image?\nASSISTANT: The image features a stop sign on a street corner"
    ```"""

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    vision_feature_layer = (
        vision_feature_layer if vision_feature_layer is not None else self.config.vision_feature_layer
    )
    vision_feature_select_strategy = (
        vision_feature_select_strategy
        if vision_feature_select_strategy is not None
        else self.config.vision_feature_select_strategy
    )

    if inputs_embeds is None:
        # 1. Extra the input embeddings
        inputs_embeds = self.get_input_embeddings()(input_ids)

        # 2. Merge text and images
        if pixel_values is not None and input_ids.shape[1] != 1:
            image_outputs = self.vision_tower(pixel_values, output_hidden_states=True)
            # this is not memory efficient at all (output_hidden_states=True) will save all the hidden stated.
            selected_image_feature = image_outputs.hidden_states[vision_feature_layer]

            if vision_feature_select_strategy == "default":
                selected_image_feature = selected_image_feature[:, 1:]
            elif vision_feature_select_strategy == "full":
                selected_image_feature = selected_image_feature
            else:
                raise ValueError(
                    f"Unexpected select feature strategy: {self.config.vision_feature_select_strategy}"
                )

            image_features = self.multi_modal_projector(selected_image_feature)
            inputs_embeds, attention_mask, labels, position_ids = self._merge_input_ids_with_image_features(
                image_features, inputs_embeds, input_ids, attention_mask, labels
            )
            if labels is None:
                labels = torch.full_like(attention_mask, self.config.ignore_index).to(torch.long)
        else:
            # In case input_ids.shape[1] == 1 & pixel_values==None & past_key_values != None, we are in the case of
            # generation with cache
            if past_key_values is not None and pixel_values is not None and input_ids.shape[1] == 1:
                # Retrieve the first layer to inspect the logits and mask out the hidden states
                # that are set to 0
                first_layer_past_key_value = past_key_values[0][0][:, :, :, 0]

                # Sum all dimensions of head_dim (-2) to avoid random errors such as: https://github.com/huggingface/transformers/pull/28032#issuecomment-1863691941
                batch_index, non_attended_tokens = torch.where(first_layer_past_key_value.float().sum(-2) == 0)

                # Get the target length
                target_seqlen = first_layer_past_key_value.shape[-1] + 1

                extended_attention_mask = torch.ones(
                    (attention_mask.shape[0], target_seqlen - attention_mask.shape[1]),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )

                # Filter out only the tokens that can be un-attended, this can happen
                # if one uses Llava + Fused modules where the cache on the
                # first iteration is already big enough, or if one passes custom cache
                valid_indices = non_attended_tokens < extended_attention_mask.size(-1)
                new_batch_index = batch_index[valid_indices]
                new_non_attended_tokens = non_attended_tokens[valid_indices]

                # Zero-out the places where we don't need to attend
                extended_attention_mask[new_batch_index, new_non_attended_tokens] = 0

                attention_mask = torch.cat((attention_mask, extended_attention_mask), dim=1)
                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
    if mask_idx is not None and past_key_values is None:
        # top-k masking
        # for att_mask, idx in zip(attention_mask, mask_idx):
        #     att_mask[idx] = 0

        #token noising    
        for input_embed, idx in zip(inputs_embeds, mask_idx):
            # print(input_ids.shape[-1])
            # print(inputs_embeds.shape)
            # print(input_embed.shape)
            # print(mask_idx)
            plus = input_ids_length
            # input_embed[idx] = torch.randn(input_embed[idx].size(), dtype=input_embed.dtype).to(input_embed.device) * 0.1
            #input_embed[idx] = add_diffusion_noise(input_embed[idx], noise_step=500)
            if masking_scheme.lower() == "ones":
                input_embed[idx +  plus] = 1.0
                # print("ones")
            elif masking_scheme.lower() == "zeros":
                input_embed[idx +  plus] = 0.0
                # print("zeros")
            elif masking_scheme.lower() == "noise":
                input_embed[idx +  plus] = torch.randn(input_embed[idx + plus].size(), dtype=input_embed.dtype).to(input_embed.device)
                # print("noise")
            else:
                input_embed[idx +  plus] = 0.0
    outputs = self.language_model(
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    logits = outputs[0]

    loss = None
    if labels is not None:
        # Shift so that tokens < n predict n
        if attention_mask is not None:
            shift_attention_mask = attention_mask[..., 1:]
            shift_logits = logits[..., :-1, :][shift_attention_mask.to(logits.device) != 0].contiguous()
            shift_labels = labels[..., 1:][shift_attention_mask.to(labels.device) != 0].contiguous()
        else:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1).to(shift_logits.device)
        )

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return LlavaCausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )
InternLM2_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or
            when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, decoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""
@add_start_docstrings_to_model_forward(InternLM2_INPUTS_DOCSTRING)
@replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
def internvl_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    mask_idx: Optional[torch.Tensor] = None,
    masking_scheme=None,
    input_ids_length = 10,
) -> Union[Tuple, CausalLMOutputWithPast]:
    r"""
    Args:
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

    Returns:

    Example:

    ```python
    >>> from transformers import AutoTokenizer, InternLM2ForCausalLM

    >>> model = InternLM2ForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
    >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

    >>> prompt = "Hey, are you conscious? Can you talk to me?"
    >>> inputs = tokenizer(prompt, return_tensors="pt")

    >>> # Generate
    >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
    >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
    ```"""

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    if mask_idx is not None and past_key_values is None:
        # top-k masking
        # for att_mask, idx in zip(attention_mask, mask_idx):
        #     att_mask[idx] = 0

        #token noising    
        for input_embed, idx in zip(inputs_embeds, mask_idx):
            # print(input_ids.shape[-1])
            # print(inputs_embeds.shape)
            # print(input_embed.shape)
            # print(mask_idx)
            plus = input_ids_length
            # input_embed[idx] = torch.randn(input_embed[idx].size(), dtype=input_embed.dtype).to(input_embed.device) * 0.1
            #input_embed[idx] = add_diffusion_noise(input_embed[idx], noise_step=500)
            if masking_scheme.lower() == "ones":
                input_embed[idx +  plus] = 1.0
                # print("ones")
            elif masking_scheme.lower() == "zeros":
                input_embed[idx +  plus] = 0.0
                # print("zeros")
            elif masking_scheme.lower() == "noise":
                input_embed[idx +  plus] = torch.randn(input_embed[idx + plus].size(), dtype=input_embed.dtype).to(input_embed.device)
                # print("noise")
            else:
                input_embed[idx +  plus] = 0.0
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    hidden_states = outputs[0]
    logits = self.output(hidden_states)
    logits = logits.float()

    loss = None
    if labels is not None:
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    device = input_ids.device if input_ids is not None else inputs_embeds.device
    output = CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )
    output['logits'] = output['logits'].to(device)
    return output

def sample(
    self,
    input_ids: torch.LongTensor,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    logits_warper: Optional[LogitsProcessorList] = None,
    max_length: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[Union[int, List[int]]] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_scores: Optional[bool] = None,
    return_dict_in_generate: Optional[bool] = None,
    synced_gpus: bool = False,
    streamer: Optional["BaseStreamer"] = None,
    use_avisc : Optional[bool] = True,
    use_m3id : Optional[bool] = False,
    **model_kwargs,
) -> Union[SampleOutput, torch.LongTensor]:
    # init values
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    if max_length is not None:
        warnings.warn(
            "`max_length` is deprecated in this function, use"
            " `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
            UserWarning,
        )
        stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
    logits_warper = logits_warper if logits_warper is not None else LogitsProcessorList()
    pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id

    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
    output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
    output_attentions = (
        output_attentions if output_attentions is not None else self.generation_config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
    )

    return_dict_in_generate = (
        return_dict_in_generate
        if return_dict_in_generate is not None
        else self.generation_config.return_dict_in_generate
    )

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        )

    # keep track of which sequences are already finished
    unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)

    this_peer_finished = False  # used by synced_gpus only

    # auto-regressive generation
    model_kwargs_method = copy.deepcopy(model_kwargs)
    model_kwargs_cd = copy.deepcopy(model_kwargs)
    model_kwargs_m3id = copy.deepcopy(model_kwargs)
    t = 1
    while True:
        if synced_gpus:
            # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
            # The following logic allows an early break if all peers finished generating their sequence
            this_peer_finished_flag = torch.tesnsor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
            # send 0.0 if we finished, 1.0 otherwise
            dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
            # did all peers finish? the reduced sum will be 0.0 then
            if this_peer_finished_flag.item() == 0.0:
                break


        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
        output_attentions = use_avisc and not (model_kwargs.get("use_cache") and model_kwargs.get("past_key_values") is not None)

        # forward pass to get next token
        outputs = self(
            **model_inputs,
            return_dict=True,
            output_attentions=output_attentions, 
            output_hidden_states=output_hidden_states, 
        )

        if synced_gpus and this_peer_finished:
            continue  # don't waste resources running the code we don't need

        next_token_logits = outputs.logits[:, -1, :]
        
        ## For contrastive decoding initial
        use_cd = model_kwargs.get("images_cd") != None
        output_attentions_wo_img = (
            output_attentions if output_attentions is not None else self.generation_config.output_attentions
        )
        output_hidden_states_wo_img = (
            output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
        )
        if use_avisc:
            
            ## analyzing attetion logit
            layer_gamma = model_kwargs.get("layer_gamma")
            masking_scheme = model_kwargs.get("masking_scheme")
            lamb = model_kwargs.get("lamb")
            model_name = model_kwargs.get("model_name") if model_kwargs.get("model_name") is not None else "blip" 
            input_ids_length = model_kwargs.get("input_ids_length") if model_kwargs.get("input_ids_length") is not None else 10
            def count_top_p(img_att_logit, top_p=0.8):
                    """
                    img_att_logit: torch.Tensor, shape (1, N)
                    """
                    norm_img_att_logit = img_att_logit / img_att_logit.sum()
                    sorted_img_att_logit = torch.sort(norm_img_att_logit, descending=True)[0]
                    
                    return (torch.cumsum(sorted_img_att_logit, dim=1) < top_p).sum() + 1
            
            
            #model_inputs_method = self.prepare_inputs_for_generation_method(input_ids, **model_kwargs_method)
            model_inputs_method = self.prepare_inputs_for_generation(input_ids, **model_kwargs_method)
            mask_idx = None
            if model_inputs_method.get("past_key_values") is None:
                attention = outputs.attentions             
                
                if model_name.lower() == "llava":
                    
                    st = input_ids_length
                    ed = st + 576
                    #img_idx = list(range(35, 35+576))
                    
                    img_idx = list(range(20,576))

                elif model_name.lower() == "blip":

                    img_idx = list(range(32))
                
                layer_img_att_portion = []
                for logit in outputs.attentions:
                    #print(logit.shape)
                    img_logit = logit.mean(dim=1)[:,-1, img_idx]
                    layer_img_att_portion.append(img_logit.sum())
                    
                layer_img_att_portion = torch.stack(layer_img_att_portion, dim=0)
                total_img_att_portion = layer_img_att_portion.sum()
                layer_img_att_portion = layer_img_att_portion / total_img_att_portion
                k = count_top_p(layer_img_att_portion.unsqueeze(0), top_p=float(layer_gamma))
                
                _, top_k_lay_idx = torch.topk(layer_img_att_portion.float(), k, dim=0)
                
            
                #######
                # Thresholding
                att_logits = torch.stack([attention[i].mean(dim=1)[:,-1,img_idx] for i in top_k_lay_idx], dim=1)  # [batch_size, num_layer, seq_len]
                img_att_logits = att_logits.mean(dim=1)
                
                # except global context token masking
                mask_idx = torch.where(img_att_logits < img_att_logits.mean() + img_att_logits.std() * lamb)[1].unsqueeze(0)

                
            output_attentions_method = False
            model_inputs_method.update(
                {
                    "mask_idx": mask_idx,
                    "masking_scheme": masking_scheme
                }
            )

            
            outputs_method = self(
                self,
                **model_inputs_method,
                return_dict=True,
                output_attentions=output_attentions_method,
                output_hidden_states=output_hidden_states,
            )
            
            next_token_logits_method = outputs_method.logits[:, -1, :]
            
            if torch.isnan(next_token_logits_method).any():
                next_token_logits_method = next_token_logits   
            
            avisc_alpha = model_kwargs.get("cd_alpha") if model_kwargs.get("cd_alpha") is not None else 1.0
            avisc_beta = model_kwargs.get("cd_beta") if model_kwargs.get("cd_beta") is not None else 0.1
            
            cutoff = torch.log(torch.tensor(avisc_beta)) + next_token_logits.max(dim=-1, keepdim=True).values
            diffs = (1+avisc_alpha)*next_token_logits - avisc_alpha*next_token_logits_method
            avisc_logits = diffs.masked_fill(next_token_logits < cutoff, -float("inf"))
            avisc_logits = logits_processor(input_ids, avisc_logits)
            avisc_logits = logits_warper(input_ids, avisc_logits)

            next_token_scores = avisc_logits
            avisc_probs = nn.functional.softmax(avisc_logits, dim=-1)
            next_tokens = torch.multinomial(avisc_probs, num_samples=1).squeeze(1)
        
        elif use_cd:
            ## cd_comments: forward pass of the model with distorted image input
            model_inputs_cd = self.prepare_inputs_for_generation_cd(input_ids, **model_kwargs_cd)
            outputs_cd = self(
                **model_inputs_cd,
                return_dict=True,
                output_attentions=output_attentions_wo_img,
                output_hidden_states=output_hidden_states_wo_img,
            )
            next_token_logits_cd = outputs_cd.logits[:, -1, :]
            
            ## cd_comments: pre-process logits from contrastive inputs
            cd_alpha = model_kwargs.get("cd_alpha") if model_kwargs.get("cd_alpha") is not None else 0.5
            cd_alpha = 1.0
            cd_beta = model_kwargs.get("cd_beta") if model_kwargs.get("cd_beta") is not None else 0.1

            # version 2 set cutoff for Adaptive Plausibility Constraints
            cutoff = torch.log(torch.tensor(cd_beta)) + next_token_logits.max(dim=-1, keepdim=True).values
            
            diffs = (1+cd_alpha)*next_token_logits - cd_alpha*next_token_logits_cd
            cd_logits = diffs.masked_fill(next_token_logits < cutoff, -float("inf"))

            ## cd_comments: apply temperature warping and top-k filtering in contrastive decoding
            cd_logits = logits_processor(input_ids, cd_logits)
            cd_logits = logits_warper(input_ids, cd_logits)

            next_token_scores = cd_logits
            cd_probs = nn.functional.softmax(cd_logits, dim=-1)
            next_tokens = torch.multinomial(cd_probs, num_samples=1).squeeze(1)

        elif use_m3id:
            import math
            lamda = 0.02
            gamma_t = math.exp(-lamda * t)
            t += 1
            
            model_inputs_m3id = self.prepare_inputs_for_generation_m3id(input_ids, **model_kwargs_m3id)
            outputs_m3id = self(
                **model_inputs_m3id,
                return_dict=True,
                output_attentions=output_attentions_wo_img,
                output_hidden_states=output_hidden_states_wo_img,
            )
            next_token_logits_m3id = outputs_m3id.logits[:, -1, :]
            
            cd_beta = model_kwargs.get("cd_beta") if model_kwargs.get("cd_beta") is not None else 0.1

            # version 2 set cutoff for Adaptive Plausibility Constraints
            cutoff = torch.log(torch.tensor(cd_beta)) + next_token_logits.max(dim=-1, keepdim=True).values
            
            lc = torch.log_softmax(next_token_logits, dim=-1)
            lu = torch.log_softmax(next_token_logits_m3id, dim=-1)
            m3id_logit = lc + ((1-gamma_t)/gamma_t)*(lc - lu)
            m3id_logit = m3id_logit.masked_fill(next_token_logits < cutoff, -float("inf"))
            
            
            m3id_logit = logits_processor(input_ids, m3id_logit)
            m3id_logit = logits_warper(input_ids, m3id_logit)
            
            next_token_scores = m3id_logit
            m3id_probs = nn.functional.softmax(m3id_logit, dim=-1)
            next_tokens = torch.multinomial(m3id_probs, num_samples=1).squeeze(1)
        else:
            next_token_scores = logits_processor(input_ids, next_token_logits)
            next_token_scores = logits_warper(input_ids, next_token_scores)
            next_token_scores = next_token_scores
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            
        # Store scores, attentions and hidden_states when required
        if return_dict_in_generate:
            if output_scores:
                scores += (next_token_scores,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                )
                if self.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)

            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if self.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )


        # finished sentences should have their next token be a padding token
        if eos_token_id is not None:
            if pad_token_id is None:
                raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        if streamer is not None:
            streamer.put(next_tokens.cpu())
        model_kwargs = self._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
        )
        if use_avisc:
            model_kwargs_method = self._update_model_kwargs_for_generation(
                outputs_method, model_kwargs_method, is_encoder_decoder=self.config.is_encoder_decoder
            )
        ## cd_comments: update model_kwargs_cd for contrastive decoding
        if use_cd:
            if use_avisc:
                model_kwargs_cd = model_inputs_method
            else:
                model_kwargs_cd = self._update_model_kwargs_for_generation(
                    outputs_cd, model_kwargs_cd, is_encoder_decoder=self.config.is_encoder_decoder
                )
        if use_m3id:
            model_kwargs_cd = self._update_model_kwargs_for_generation(
                outputs_m3id, model_kwargs_m3id, is_encoder_decoder=self.config.is_encoder_decoder
            )

        # if eos_token was found in one sentence, set sentence to finished
        if eos_token_id_tensor is not None:
            unfinished_sequences = unfinished_sequences.mul(
                next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
            )

            # stop when each sentence is finished
            if unfinished_sequences.max() == 0:
                this_peer_finished = True

        # stop if we exceed the maximum length
        if stopping_criteria(input_ids, scores):
            this_peer_finished = True

        if this_peer_finished and not synced_gpus:
            break

    if streamer is not None:
        streamer.end()

    if return_dict_in_generate:
        if self.config.is_encoder_decoder:
            return SampleEncoderDecoderOutput(
                sequences=input_ids,
                scores=scores,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
            )
        else:
            return SampleDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
            )
    else:
        return input_ids
def _validate_model_kwargs(self, model_kwargs: Dict[str, Any]):
    """Validates model kwargs for generation. Generate argument typos will also be caught here."""
    # Excludes arguments that are handled before calling any model function
    if self.config.is_encoder_decoder:
        for key in ["decoder_input_ids"]:
            model_kwargs.pop(key, None)

    unused_model_args = []
    model_args = set(inspect.signature(
        self.prepare_inputs_for_generation).parameters)
    # `kwargs`/`model_kwargs` is often used to handle optional forward pass inputs like `attention_mask`. If
    # `prepare_inputs_for_generation` doesn't accept them, then a stricter check can be made ;)
    if "kwargs" in model_args or "model_kwargs" in model_args:
        model_args |= set(inspect.signature(self.forward).parameters)

    # Encoder-Decoder models may also need Encoder arguments from `model_kwargs`
    if self.config.is_encoder_decoder:
        base_model = getattr(self, self.base_model_prefix, None)

        # allow encoder kwargs
        encoder = getattr(self, "encoder", None)
        # `MusicgenForConditionalGeneration` has `text_encoder` and `audio_encoder`.
        # Also, it has `base_model_prefix = "encoder_decoder"` but there is no `self.encoder_decoder`
        # TODO: A better way to handle this.
        if encoder is None and base_model is not None:
            encoder = getattr(base_model, "encoder", None)

        if encoder is not None:
            encoder_model_args = set(
                inspect.signature(encoder.forward).parameters)
            model_args |= encoder_model_args

        # allow decoder kwargs
        decoder = getattr(self, "decoder", None)
        if decoder is None and base_model is not None:
            decoder = getattr(base_model, "decoder", None)

        if decoder is not None:
            decoder_model_args = set(
                inspect.signature(decoder.forward).parameters)
            model_args |= {f"decoder_{x}" for x in decoder_model_args}

    for key, value in model_kwargs.items():
        if value is not None and key not in model_args:
            unused_model_args.append(key)




def greedy_search(
    self,
    input_ids: torch.LongTensor,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    max_length: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[Union[int, List[int]]] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_scores: Optional[bool] = None,
    output_logits: Optional[bool] = None,
    return_dict_in_generate: Optional[bool] = None,
    synced_gpus: bool = False,
    streamer: Optional["BaseStreamer"] = None,
    **model_kwargs,
) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
    r"""
    Generates sequences of token ids for models with a language modeling head using **greedy decoding** and can be
    used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

    <Tip warning={true}>

    In most cases, you do not need to call [`~generation.GenerationMixin.greedy_search`] directly. Use generate()
    instead. For an overview of generation strategies and code examples, check the [following
    guide](../generation_strategies).

    </Tip>


    Parameters:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            The sequence used as a prompt for the generation.
        logits_processor (`LogitsProcessorList`, *optional*):
            An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
            used to modify the prediction scores of the language modeling head applied at each generation step.
        stopping_criteria (`StoppingCriteriaList`, *optional*):
            An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
            used to tell if the generation loop should stop.

        max_length (`int`, *optional*, defaults to 20):
            **DEPRECATED**. Use `logits_processor` or `stopping_criteria` directly to cap the number of generated
            tokens. The maximum length of the sequence to be generated.
        pad_token_id (`int`, *optional*):
            The id of the *padding* token.
        eos_token_id (`Union[int, List[int]]`, *optional*):
            The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
        output_attentions (`bool`, *optional*, defaults to `False`):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under
            returned tensors for more details.
        output_hidden_states (`bool`, *optional*, defaults to `False`):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
            for more details.
        output_scores (`bool`, *optional*, defaults to `False`):
            Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
        output_logits (`bool`, *optional*, defaults to `False`):
            Whether or not to return the raw prediction logit scores. See `logits` under returned tensors
            for more details.
        return_dict_in_generate (`bool`, *optional*, defaults to `False`):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        synced_gpus (`bool`, *optional*, defaults to `False`):
            Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
        streamer (`BaseStreamer`, *optional*):
            Streamer object that will be used to stream the generated sequences. Generated tokens are passed
            through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
        model_kwargs:
            Additional model specific keyword arguments will be forwarded to the `forward` function of the model.
            If model is an encoder-decoder model the kwargs should include `encoder_outputs`.

    Return:
        [`~generation.GenerateDecoderOnlyOutput`], [`~generation.GenerateEncoderDecoderOutput`] or
        `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
        [`~generation.GenerateDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
        `return_dict_in_generate=True` or a [`~generation.GenerateEncoderDecoderOutput`] if
        `model.config.is_encoder_decoder=True`.

    Examples:

    ```python
    >>> from transformers import (
    ...     AutoTokenizer,
    ...     AutoModelForCausalLM,
    ...     LogitsProcessorList,
    ...     MinLengthLogitsProcessor,
    ...     StoppingCriteriaList,
    ...     MaxLengthCriteria,
    ... )

    >>> tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    >>> model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")

    >>> # set pad_token_id to eos_token_id because GPT2 does not have a PAD token
    >>> model.generation_config.pad_token_id = model.generation_config.eos_token_id

    >>> input_prompt = "It might be possible to"
    >>> input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids

    >>> # instantiate logits processors
    >>> logits_processor = LogitsProcessorList(
    ...     [
    ...         MinLengthLogitsProcessor(10, eos_token_id=model.generation_config.eos_token_id),
    ...     ]
    ... )
    >>> stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=20)])

    >>> outputs = model.greedy_search(
    ...     input_ids, logits_processor=logits_processor, stopping_criteria=stopping_criteria
    ... )

    >>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
    ["It might be possible to get a better understanding of the nature of the problem, but it's not"]
    ```"""


    # init values
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    if max_length is not None:
        warnings.warn(
            "`max_length` is deprecated in this function, use"
            " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
            UserWarning,
        )
        stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
    pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
    output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
    output_attentions = (
        output_attentions if output_attentions is not None else self.generation_config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
    )
    return_dict_in_generate = (
        return_dict_in_generate
        if return_dict_in_generate is not None
        else self.generation_config.return_dict_in_generate
    )

    # init attention / hidden states / scores tuples
    raw_logits = () if (return_dict_in_generate and output_logits) else None
    scores = () if (return_dict_in_generate and output_scores) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        )

    # keep track of which sequences are already finished
    unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)
    
    this_peer_finished = False  # used by synced_gpus only
    use_cd = model_kwargs.get("images_cd") != None
    #if use_cd:
        #self.prepare_inputs_for_generation_cd = prepare_inputs_for_generation_cd
    #print(use_cd)
    if use_cd:
        model_kwargs_cd = model_kwargs.copy()
        model_kwargs_cd['pixel_values'] =  model_kwargs_cd['images_cd']
    while True:
        if synced_gpus:
            # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
            # The following logic allows an early break if all peers finished generating their sequence
            this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
            # send 0.0 if we finished, 1.0 otherwise
            dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
            # did all peers finish? the reduced sum will be 0.0 then
            if this_peer_finished_flag.item() == 0.0:
                break

        # prepare model inputs
        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
        #print(model_inputs.keys())
        # forward pass to get next token
        outputs = self(
            **model_inputs,
            return_dict=True,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        #print('model_input:{}'.format(model_inputs['inputs_embeds'].shape))
        if synced_gpus and this_peer_finished:
            continue  # don't waste resources running the code we don't need

        next_token_logits = outputs.logits[:, -1, :]

        # pre-process distribution
        next_tokens_scores = logits_processor(input_ids, next_token_logits)
        #print(input_ids.shape)
        if use_cd: 
            #print(input_ids.shape)
            #print('model_input pixel_values:{}'.format(model_kwargs_cd['pixel_values'].shape))
            model_inputs_cd = self.prepare_inputs_for_generation(input_ids, **model_kwargs_cd)
            #print('model_input_cd:{}'.format(model_inputs_cd['inputs_embeds'].shape))
            outputs_cd = self(
                **model_inputs_cd,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            next_token_logits_cd = outputs_cd.logits[:, -1, :]
            

            # cd config
            cd_alpha = model_kwargs.get("cd_alpha") if model_kwargs.get("cd_alpha") is not None else 0.5
            cd_beta = model_kwargs.get("cd_beta") if model_kwargs.get("cd_beta") is not None else 0.1
            cutoff = torch.log(torch.tensor(cd_beta)) + next_token_logits.max(dim=-1, keepdim=True).values
            diffs = (1+cd_alpha)*next_token_logits - cd_alpha*next_token_logits_cd
            cd_logits = diffs.masked_fill(next_token_logits < cutoff, -float("inf"))
            cd_logits = logits_processor(input_ids, cd_logits)
            next_tokens_scores = cd_logits
            


        # Store scores, attentions and hidden_states when required
        if return_dict_in_generate:
            if output_scores:
                scores += (next_tokens_scores,)
            if output_logits:
                raw_logits += (next_token_logits,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                )
                if self.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)

            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if self.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )

        # argmax
        next_tokens = torch.argmax(next_tokens_scores, dim=-1)

        # finished sentences should have their next token be a padding token
        if eos_token_id is not None:
            if pad_token_id is None:
                raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        
        if streamer is not None:
            streamer.put(next_tokens.cpu())
        model_kwargs = self._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
        )
        if use_cd:
            model_kwargs_cd = self._update_model_kwargs_for_generation(
                outputs_cd, model_kwargs_cd, is_encoder_decoder=self.config.is_encoder_decoder
            )
        # if eos_token was found in one sentence, set sentence to finished
        if eos_token_id_tensor is not None:
            unfinished_sequences = unfinished_sequences.mul(
                next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
            )

            # stop when each sentence is finished
            if unfinished_sequences.max() == 0:
                this_peer_finished = True

        # stop if we exceed the maximum length
        if stopping_criteria(input_ids, scores):
            this_peer_finished = True

        if this_peer_finished and not synced_gpus:
            break

    if streamer is not None:
        streamer.end()

    if return_dict_in_generate:
        if self.config.is_encoder_decoder:
            return GenerateEncoderDecoderOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
        else:
            return GenerateDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
    else:
        return input_ids

from transformers.models.instructblip.modeling_instructblip import INSTRUCTBLIP_INPUTS_DOCSTRING,InstructBlipForConditionalGenerationModelOutput
from transformers.models.instructblip.configuration_instructblip import InstructBlipVisionConfig
# @add_start_docstrings_to_model_forward(INSTRUCTBLIP_INPUTS_DOCSTRING)
# @replace_return_docstrings(
#     output_type=InstructBlipForConditionalGenerationModelOutput, config_class=InstructBlipVisionConfig
# )
# def blip2_forward(
#     self,
#     pixel_values: torch.FloatTensor,
#     qformer_input_ids: torch.FloatTensor,
#     qformer_attention_mask: Optional[torch.LongTensor] = None,
#     input_ids: Optional[torch.FloatTensor] = None,
#     attention_mask: Optional[torch.LongTensor] = None,
#     decoder_input_ids: Optional[torch.LongTensor] = None,
#     decoder_attention_mask: Optional[torch.LongTensor] = None,
#     output_attentions: Optional[bool] = None,
#     output_hidden_states: Optional[bool] = None,
#     labels: Optional[torch.LongTensor] = None,
#     return_dict: Optional[bool] = None,
#     masking_scheme:Optional[str]=None,
#     #lamb:Optional[int]=None,
#     #model_name:Optional[str]=None,
#     mask_idx: Optional[torch.Tensor] = None,
# ) -> Union[Tuple, InstructBlipForConditionalGenerationModelOutput]:
#     r"""
#     labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
#         Labels for computing the language modeling loss. Indices should be in `[-100, 0, ..., config.vocab_size -
#         1]`. All labels set to `-100` are ignored (masked), the loss is only computed for labels in `[0, ...,
#         config.vocab_size]`

#     Returns:

#     Examples:

#     ```python
#     >>> from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
#     >>> import torch
#     >>> from PIL import Image
#     >>> import requests

#     >>> model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-7b")
#     >>> processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")

#     >>> device = "cuda" if torch.cuda.is_available() else "cpu"
#     >>> model.to(device)  # doctest: +IGNORE_RESULT

#     >>> url = "https://raw.githubusercontent.com/salesforce/LAVIS/main/docs/_static/Confusing-Pictures.jpg"
#     >>> image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
#     >>> prompt = "What is unusual about this image?"
#     >>> inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

#     >>> outputs = model.generate(
#     ...     **inputs,
#     ...     do_sample=False,
#     ...     num_beams=5,
#     ...     max_length=256,
#     ...     min_length=1,
#     ...     top_p=0.9,
#     ...     repetition_penalty=1.5,
#     ...     length_penalty=1.0,
#     ...     temperature=1,
#     ... )
#     >>> generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
#     >>> print(generated_text)
#     The unusual aspect of this image is that a man is ironing clothes on the back of a yellow SUV, which is parked in the middle of a busy city street. This is an unconventional approach to ironing clothes, as it requires the man to balance himself and his ironing equipment on top of the vehicle while navigating through traffic. Additionally, the presence of taxis and other vehicles in the scene further emphasizes the unusual nature of this situation.
#     ```"""
#     return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#     # step 1: forward the images through the vision encoder,
#     # to get image embeddings of shape (batch_size, seq_len, hidden_size)
#     vision_outputs = self.vision_model(
#         pixel_values=pixel_values,
#         output_attentions=output_attentions,
#         output_hidden_states=output_hidden_states,
#         return_dict=return_dict,
#     )
#     image_embeds = vision_outputs[0]

#     # step 2: forward the query tokens through the QFormer, using the image embeddings for cross-attention
#     image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)

#     # difference with BLIP-2 here: we also feed the instruction prompt to the Q-Former
#     query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
#     query_attention_mask = torch.ones(query_tokens.size()[:-1], dtype=torch.long, device=image_embeds.device)
#     if qformer_attention_mask is None:
#         qformer_attention_mask = torch.ones_like(qformer_input_ids)
#     qformer_attention_mask = torch.cat([query_attention_mask, qformer_attention_mask], dim=1)
#     query_outputs = self.qformer(
#         input_ids=qformer_input_ids,
#         attention_mask=qformer_attention_mask,
#         query_embeds=query_tokens,
#         encoder_hidden_states=image_embeds,
#         encoder_attention_mask=image_attention_mask,
#         output_attentions=output_attentions,
#         output_hidden_states=output_hidden_states,
#         return_dict=return_dict,
#     )
#     query_output = query_outputs[0][:, : query_tokens.size(1), :]

#     # step 3: use the language model, conditioned on the query outputs and the prompt
#     language_model_inputs = self.language_projection(query_output)
#     language_model_attention_mask = torch.ones(
#         language_model_inputs.size()[:-1], dtype=torch.long, device=language_model_inputs.device
#     )

#     inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

#     inputs_embeds = torch.cat([language_model_inputs, inputs_embeds.to(language_model_inputs.device)], dim=1)

#     if attention_mask is None:
#         attention_mask = torch.ones_like(input_ids)
#     attention_mask = torch.cat([language_model_attention_mask.to(attention_mask.device), attention_mask], dim=1)
#     if mask_idx is not None:
#         for input_embed, idx in zip(inputs_embeds, mask_idx):
#             if masking_scheme.lower() == "ones":
#                 input_embed[idx] = 1.0
#                 # print("ones")
#             elif masking_scheme.lower() == "zeros":
#                 input_embed[idx] = 0.0
#                 # print("zeros")
#             elif masking_scheme.lower() == "noise":
#                 input_embed[idx] = torch.randn(input_embed[idx].size(), dtype=input_embed.dtype).to(input_embed.device)
#                 # print("noise")
#             else:
#                 input_embed[idx] = 0.0
#     if self.config.use_decoder_only_language_model:
#         outputs = self.language_model(
#             inputs_embeds=inputs_embeds,
#             attention_mask=attention_mask,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )
#         logits = outputs.logits if return_dict else outputs[0]
#         loss = None
#         # we compute the loss here since we need to take into account the sequence length of the query embeds
#         if labels is not None:
#             labels = labels.to(logits.device)
#             logits = logits[:, -labels.size(1) :, :]
#             # Shift so that tokens < n predict n
#             shift_logits = logits[..., :-1, :].contiguous()
#             shift_labels = labels[..., 1:].contiguous().to(logits.device)

#             # Flatten the tokens
#             loss_fct = CrossEntropyLoss(reduction="mean")

#             loss = loss_fct(shift_logits.view(-1, self.config.text_config.vocab_size), shift_labels.view(-1))
#     else:
#         outputs = self.language_model(
#             inputs_embeds=inputs_embeds,
#             attention_mask=attention_mask,
#             decoder_input_ids=decoder_input_ids,
#             decoder_attention_mask=decoder_attention_mask,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#             labels=labels,
#         )
#         loss = outputs.loss if return_dict else outputs[0]
#         logits = outputs.logits if return_dict else outputs[1]

#     if not return_dict:
#         output = (logits, vision_outputs, query_outputs, outputs)
#         return ((loss,) + output) if loss is not None else output

#     return InstructBlipForConditionalGenerationModelOutput(
#         loss=loss,
#         logits=logits,
#         vision_outputs=vision_outputs,
#         qformer_outputs=query_outputs,
#         language_model_outputs=outputs,
#     )

from transformers.models.llama.modeling_llama import LLAMA_INPUTS_DOCSTRING,CausalLMOutputWithPast

@add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
@replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
def blip2_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    masking_scheme:Optional[str]=None,
        
    mask_idx: Optional[torch.Tensor] = None,
) -> Union[Tuple, CausalLMOutputWithPast]:
    r"""
    Args:
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

    Returns:

    Example:

    ```python
    >>> from transformers import AutoTokenizer, LlamaForCausalLM

    >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
    >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

    >>> prompt = "Hey, are you conscious? Can you talk to me?"
    >>> inputs = tokenizer(prompt, return_tensors="pt")

    >>> # Generate
    >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
    >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
    ```"""
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    if mask_idx is not None and past_key_values is None:

        for input_embed, idx in zip(inputs_embeds, mask_idx):
            if masking_scheme.lower() == "ones":
                input_embed[idx] = 1.0
                # print("ones")
            elif masking_scheme.lower() == "zeros":
                input_embed[idx] = 0.0
                # print("zeros")
            elif masking_scheme.lower() == "noise":
                input_embed[idx] = torch.randn(input_embed[idx].size(), dtype=input_embed.dtype).to(input_embed.device)
                # print("noise")
            else:
                input_embed[idx] = 0.0
    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
    )

    hidden_states = outputs[0]
    if self.config.pretraining_tp > 1:
        lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
        logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
        logits = torch.cat(logits, dim=-1)
    else:
        logits = self.lm_head(hidden_states)
    logits = logits.float()

    loss = None
    if labels is not None:
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )
def evolve_vcd_sampling():
    transformers.generation.utils.GenerationMixin.sample = sample
    transformers.generation.utils.GenerationMixin._validate_model_kwargs = _validate_model_kwargs
    transformers.generation.utils.GenerationMixin.greedy_search = greedy_search
    transformers.models.llava.modeling_llava.LlavaForConditionalGeneration.forward = forward   
    transformers.models.llama.modeling_llama.LlamaForCausalLM.forward = blip2_forward

def evolve_vcd_sampling_true():
    #transformers.generation.utils.GenerationMixin.sample = sample
    transformers.generation.utils.GenerationMixin._validate_model_kwargs = _validate_model_kwargs
    transformers.generation.utils.GenerationMixin.greedy_search = greedy_search
    #transformers.models.llava.modeling_llava.LlavaForConditionalGeneration.forward = forward   
    #transformers.models.llama.modeling_llama.LlamaForCausalLM.forward = blip2_forward

def evolve_vcd_sampling_llava():
    transformers.generation.utils.GenerationMixin.sample = sample
    transformers.generation.utils.GenerationMixin._validate_model_kwargs = _validate_model_kwargs
    transformers.generation.utils.GenerationMixin.greedy_search = greedy_search
    transformers.models.llava.modeling_llava.LlavaForConditionalGeneration.forward = forward   
    
    #transformers.models.llama.modeling_llama.LlamaForCausalLM.forward = blip2_forward

    
def evolve_vcd_sampling_llava_true():
    #transformers.generation.utils.GenerationMixin.sample = sample
    transformers.generation.utils.GenerationMixin._validate_model_kwargs = _validate_model_kwargs
    transformers.generation.utils.GenerationMixin.greedy_search = greedy_search
    #transformers.models.llava.modeling_llava.LlavaForConditionalGeneration.forward = forward   
    #transformers.models.llama.modeling_llama.LlamaForCausalLM.forward = blip2_forward
