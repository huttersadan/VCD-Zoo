import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss


from transformers.generation.utils import SampleOutput, SampleEncoderDecoderOutput, SampleDecoderOnlyOutput
from transformers.generation.utils import GenerateNonBeamOutput
from typing import List, Optional, Union, Dict
import warnings
import torch.distributed as dist
from transformers.generation.utils import GenerateEncoderDecoderOutput
from transformers.generation.utils import GenerateDecoderOnlyOutput
import inspect
from transformers.cache_utils import Cache
import transformers
import transformers.models 



# transformers.models.instructblip.modeling_instructblip
@torch.no_grad()
def generate(
    self,
    pixel_values: torch.FloatTensor,
    qformer_input_ids: Optional[torch.LongTensor] = None,
    qformer_attention_mask: Optional[torch.LongTensor] = None,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.LongTensor] = None,

    # vcd input
    pixel_values_cd: Optional[torch.FloatTensor] = None,
    attention_mask_cd: Optional[torch.LongTensor] = None,
    # vcd config
    cd_alpha: Optional[float] = 1,
    cd_beta: Optional[float] = 0.1,


    **generate_kwargs,
) -> torch.LongTensor:
    """
    Overrides `generate` function to be able to use the model as a conditional generator.

    Args:
        pixel_values (`torch.FloatTensor` of shape (batch_size, num_channels, height, width)):
            Input images to be processed.
        qformer_input_ids (`torch.LongTensor` of shape (batch_size, sequence_length), *optional*):
            The sequence used as a prompt to be fed to the Q-Former module.
        qformer_attention_mask (`torch.LongTensor` of shape (batch_size, sequence_length), *optional*):
            Mask to avoid performing attention on padding token indices.
        input_ids (`torch.LongTensor` of shape (batch_size, sequence_length), *optional*):
            The sequence used as a prompt for the generation.
        attention_mask (`torch.LongTensor` of shape (batch_size, sequence_length), *optional*):
            Mask to avoid performing attention on padding token indices.

    Returns:
        captions (list): A list of strings of length batch_size * num_captions.
    """
    if hasattr(self, "hf_device_map"):
        # preprocess for `accelerate`
        self._preprocess_accelerate()
    #import ipdb;ipdb.set_trace()
    batch_size = pixel_values.shape[0]
    image_embeds = self.vision_model(pixel_values, return_dict=True).last_hidden_state

    image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)

    query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
    query_attention_mask = torch.ones(query_tokens.size()[:-1], dtype=torch.long, device=image_embeds.device)
    
    # vcd
    use_cd = pixel_values_cd is not None
    if use_cd:
        image_embeds_cd = self.vision_model(
            pixel_values_cd,
            return_dict=True,
        ).last_hidden_state
        image_attention_mask_cd = torch.ones(image_embeds_cd.size()[:-1], dtype=torch.long, device=image_embeds_cd.device)
        query_tokens_cd = self.query_tokens.expand(image_embeds_cd.shape[0], -1, -1)
        query_attention_mask_cd = torch.ones(query_tokens_cd.size()[:-1], dtype=torch.long, device=image_embeds_cd.device)



    if qformer_attention_mask is None:
        qformer_attention_mask = torch.ones_like(qformer_input_ids)

    if use_cd:
       qformer_attention_mask_cd = torch.cat([query_attention_mask_cd, qformer_attention_mask], dim=1)
    qformer_attention_mask = torch.cat([query_attention_mask, qformer_attention_mask], dim=1)

    


    query_outputs = self.qformer(
        input_ids=qformer_input_ids,
        attention_mask=qformer_attention_mask,
        query_embeds=query_tokens,
        encoder_hidden_states=image_embeds,
        encoder_attention_mask=image_attention_mask,
        return_dict=True,
    )
    query_output = query_outputs.last_hidden_state[:, : query_tokens.size(1), :]

    language_model_inputs = self.language_projection(query_output)
    language_attention_mask = torch.ones(
        language_model_inputs.size()[:-1], dtype=torch.long, device=language_model_inputs.device
    )

    if use_cd:
        query_outputs_cd = self.qformer(
            input_ids=qformer_input_ids,
            attention_mask=qformer_attention_mask_cd,
            query_embeds=query_tokens_cd,
            encoder_hidden_states=image_embeds_cd,
            encoder_attention_mask=image_attention_mask_cd,
            return_dict=True,
        )
        query_output_cd = query_outputs_cd.last_hidden_state[:, : query_tokens_cd.size(1), :]
        language_model_inputs_cd = self.language_projection(query_output_cd)
        language_attention_mask_cd = torch.ones(
            language_model_inputs_cd.size()[:-1], dtype=torch.long, device=language_model_inputs_cd.device
        )


    if input_ids is None:
        input_ids = (
            torch.LongTensor([[self.config.text_config.bos_token_id]])
            .repeat(batch_size, 1)
            .to(image_embeds.device)
        )
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    attention_mask = torch.cat([language_attention_mask, attention_mask.to(language_attention_mask.device)], dim=1)

    # concatenate query embeddings with prompt embeddings
    inputs_embeds = self.get_input_embeddings()(input_ids)
    inputs_embeds = torch.cat([language_model_inputs, inputs_embeds.to(language_model_inputs.device)], dim=1)

    if use_cd:
        if attention_mask_cd is None:
            attention_mask_cd = torch.ones_like(input_ids)
        attention_mask_cd = torch.cat([language_attention_mask_cd, attention_mask_cd.to(language_attention_mask_cd.device)], dim=1)
        inputs_embeds_cd = self.get_input_embeddings()(input_ids)
        inputs_embeds_cd = torch.cat([language_model_inputs_cd, inputs_embeds_cd.to(language_model_inputs_cd.device)], dim=1)
    
    if use_cd:
        outputs = self.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,

            # vcd
            inputs_embeds_cd = inputs_embeds_cd,
            attention_mask_cd = attention_mask_cd,
            cd_alpha = cd_alpha,
            cd_beta = cd_beta,

            **generate_kwargs,
        )
    else:
        outputs = self.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **generate_kwargs,
        )

    

    # the InstructBLIP authors used inconsistent tokenizer/model files during training,
    # with the tokenizer's bos token being set to </s> which has ID=2,
    # whereas the model's text config has bos token id = 0
    if self.config.text_config.architectures[0] == "LLaMAForCausalLM":
        if isinstance(outputs, torch.Tensor):
            outputs[outputs == 0] = 2
        else:
            outputs.sequences[outputs.sequences == 0] = 2

    return outputs





# transformers.models.llama.modeling_llama.LlamaForCausalLM
def prepare_inputs_for_generation_cd(
    self, input_ids, past_key_values=None, attention_mask_cd=None, inputs_embeds_cd=None, **kwargs
):
    past_length = 0
    if past_key_values is not None:
        if isinstance(past_key_values, Cache):
            cache_length = past_key_values.get_seq_length()
            past_length = past_key_values.seen_tokens
            max_cache_length = past_key_values.get_max_length()
        else:
            cache_length = past_length = past_key_values[0][0].shape[2]
            max_cache_length = None

        # Keep only the unprocessed tokens:
        # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
        # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
        # input)
        if attention_mask_cd is not None and attention_mask_cd.shape[1] > input_ids.shape[1]:
            input_ids = input_ids[:, -(attention_mask_cd.shape[1] - past_length) :]
        # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
        # input_ids based on the past_length.
        elif past_length < input_ids.shape[1]:
            input_ids = input_ids[:, past_length:]
        # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

        # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
        if (
            max_cache_length is not None
            and attention_mask_cd is not None
            and cache_length + input_ids.shape[1] > max_cache_length
        ):
            attention_mask_cd = attention_mask_cd[:, -max_cache_length:]

    position_ids = kwargs.get("position_ids", None)
    if attention_mask_cd is not None and position_ids is None:
        # create position_ids on the fly for batch generation
        position_ids = attention_mask_cd.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask_cd == 0, 1)
        if past_key_values:
            position_ids = position_ids[:, -input_ids.shape[1] :]

    if past_key_value := getattr(self.model.layers[0].self_attn, "past_key_value", None):
        # generation with static cache
        past_length = past_key_value.get_seq_length()
        input_ids = input_ids[:, past_length:]
        position_ids = position_ids[:, past_length:]

    # TODO @gante we should only keep a `cache_position` in generate, and do +=1.
    # same goes for position ids. Could also help with continued generation.
    cache_position = kwargs.get("cache_position", None)
    if cache_position is None:
        cache_position = torch.arange(
            past_length, past_length + position_ids.shape[-1], device=position_ids.device
        )

    # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
    if inputs_embeds_cd is not None and past_key_values is None:
        model_inputs = {"inputs_embeds": inputs_embeds_cd}
    else:
        model_inputs = {"input_ids": input_ids}

    model_inputs.update(
        {
            "position_ids": position_ids,
            "cache_position": cache_position,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask_cd,
        }
    )
    return model_inputs



from transformers.generation.logits_process import (
    LogitsProcessorList,
)
from transformers.generation.stopping_criteria import (
    StoppingCriteria,
    StoppingCriteriaList,
    validate_stopping_criteria,
)


# transformers.generation.utils.GenerationMixin 
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
    use_cd = model_kwargs.get("inputs_embeds_cd") != None
    #if use_cd:
        #self.prepare_inputs_for_generation_cd = prepare_inputs_for_generation_cd
    #print(use_cd)
    if use_cd:
        
        model_kwargs_cd = model_kwargs.copy()
        model_kwargs_cd['inputs_embeds'] =  model_kwargs_cd['inputs_embeds_cd']
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


  
# transformers.generation.utils.GenerationMixin 
def _validate_model_kwargs(self, model_kwargs: Dict[str, Any]):
    """Validates model kwargs for generation. Generate argument typos will also be caught here."""
    # If a `Cache` instance is passed, checks whether the model is compatible with it
    if isinstance(model_kwargs.get("past_key_values", None), Cache) and not self._supports_cache_class:
        raise ValueError(
            f"{self.__class__.__name__} does not support an instance of `Cache` as `past_key_values`. Please "
            "check the model documentation for supported cache formats."
        )

    # Excludes arguments that are handled before calling any model function
    if self.config.is_encoder_decoder:
        for key in ["decoder_input_ids"]:
            model_kwargs.pop(key, None)

    unused_model_args = []
    model_args = set(inspect.signature(self.prepare_inputs_for_generation).parameters)
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
            encoder_model_args = set(inspect.signature(encoder.forward).parameters)
            model_args |= encoder_model_args

        # allow decoder kwargs
        decoder = getattr(self, "decoder", None)
        if decoder is None and base_model is not None:
            decoder = getattr(base_model, "decoder", None)

        if decoder is not None:
            decoder_model_args = set(inspect.signature(decoder.forward).parameters)
            model_args |= {f"decoder_{x}" for x in decoder_model_args}

        # allow assistant_encoder_outputs to be passed if we're doing assisted generating
        if "assistant_encoder_outputs" in model_kwargs:
            model_args |= {"assistant_encoder_outputs"}

    for key, value in model_kwargs.items():
        if value is not None and key not in model_args:
            unused_model_args.append(key)

    # if unused_model_args:
    #     raise ValueError(
    #         f"The following `model_kwargs` are not used by the model: {unused_model_args} (note: typos in the"
    #         " generate arguments will also show up in this list)"
    #     )

def evolve_vcd_sampling_blip2():
    transformers.generation.utils.GenerationMixin._validate_model_kwargs = _validate_model_kwargs
    transformers.generation.utils.GenerationMixin.greedy_search = greedy_search
    transformers.models.instructblip.modeling_instructblip.InstructBlipForConditionalGeneration.generate = generate
    
