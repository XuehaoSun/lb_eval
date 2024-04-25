import logging
import time

import requests
from requests.exceptions import RequestException
from tqdm import tqdm

from lm_eval.api.model import LM
from lm_eval.api.registry import register_model

import llama_cpp
from llama_cpp import Llama
from typing import Iterator, List, Optional, Union, Dict


def get_result(logprobs, context_length):
    is_greedy = True
    offsets = logprobs["text_offset"]
    tokens = logprobs["tokens"]
    tokens_logprobs = logprobs["token_logprobs"]

    idx = 0
    while offsets[idx] < context_length:
        idx += 1
    continuation_logprobs = sum(tokens_logprobs[idx:-1])
    for i in range(idx, len(tokens)):
        token = tokens[i]
        top_tokens = logprobs["top_logprobs"][i]
        top_token = max(top_tokens.keys(), key=lambda x: top_tokens[x])
        if top_token != token:
            is_greedy = False
            break

    return continuation_logprobs, is_greedy


@register_model("gguf-custom", "ggml-custom")
class WrapperGGUFLM(LM):
    def __init__(self, gguf_model=None, ftype="*q4_0.gguf", max_length=2048, **kwargs):
        super().__init__()
        assert gguf_model, "must pass `base_url` to use GGUF LM!"

        self.model = Llama.from_pretrained(
                repo_id=gguf_model,
                filename=ftype,
                logits_all=True,
                n_ctx=2048,
                n_threads=16,
                use_mlock=True,
                n_threads_batch=24,
                mul_mat_q=True,
                offload_kqv=True,
                last_n_tokens_size=64,
                cache=False,
                cache_type='ram',
                verbose=True,)

        self.logprobs = 10
        self.temperature = 0.0
        self.max_length = max_length

    def gguf_completion(
        self, context, continuation=None, stop=None, retries=3, delay=5, **kwargs
    ):
        for _ in range(retries):
            try:
                prompt = context
                request = {
                    "prompt": prompt,
                    "logprobs": self.logprobs,
                    "temperature": self.temperature,
                }
                if continuation:
                    prompt += continuation
                    request.update({"prompt": prompt, "max_tokens": 1, "echo": True})
                if stop is not None:
                    request["stop"] = stop
                response = self.model(**request)
                return response
            except RequestException as e:
                logger.error(f"RequestException: {e}")
                time.sleep(delay)  # wait before retrying
        else:
            raise Exception(f"Failed to get a valid response after {retries} retries.")

    def loglikelihood(self, requests, disable_tqdm: bool = False):
        if not requests:
            return []
        res = []
        for context, continuation in tqdm(
            [req.args for req in requests], disable=disable_tqdm
        ):
            response = self.gguf_completion(context=context, continuation=continuation)
            if response and "choices" in response and response["choices"]:
                choice = response["choices"][0]
                logprobs = choice.get("logprobs")
                if (
                    logprobs
                    and "token_logprobs" in logprobs
                    and logprobs["token_logprobs"]
                ):
                    logprob, is_greedy = get_result(logprobs, len(context))
                    res.append((logprob, is_greedy))
                else:
                    logger.warning(
                        "Invalid logprobs data. Expected 'logprobs' to contain 'token_logprobs' list."
                    )
            else:
                logger.error(
                    f"Invalid response for loglikelihood. Response: {response}"
                )
                assert False
        return res

    def generate_until(self, requests, disable_tqdm: bool = False):
        if not requests:
            return []

        res = []
        for request in tqdm([req.args for req in requests], disable=disable_tqdm):
            inp = request[0]
            request_args = request[1]
            until = request_args.get("until", ["</s>"])
            response = self.gguf_completion(context=inp, stop=until)
            if response and "choices" in response and response["choices"]:
                choice = response["choices"][0]
                if "text" in choice:
                    generated_text = choice["text"].strip()
                    res.append(generated_text)
                else:
                    logger.error(
                        f"Invalid response for greedy_until. Response: {response}"
                    )
                    res.append(None)  # Add default value in case of error
            else:
                logger.error(f"Invalid response for greedy_until. Response: {response}")
                res.append(None)  # Add default value in case of error
        return res

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False):
        raise NotImplementedError(
            "loglikelihood_rolling not yet supported for GGUF models"
        )
