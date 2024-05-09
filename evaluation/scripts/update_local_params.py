import json
import re
from huggingface_hub.hf_api import ModelInfo, get_safetensors_metadata, model_info as get_model_info, get_hf_file_metadata, hf_hub_url
from huggingface_hub import hf_hub_download

# Map model IDs to the number of bytes used for one parameter. So, 4 bytes for fp32, 2 bytes for fp16, etc.
# By default, we assume that the model is stored in fp32.
KNOWN_SIZE_FACTOR = {
    "gptq": {"4bit": 8, "8bit": 4},
    "awq": {"4bit": 8},
    "bitsandbytes": {"4bit": 2}
}

BYTES = {
    "I32": 4,
    "F16": 2,
    "BF16": 2,
    "F32": 4,
    "U8": 1}

from huggingface_hub.utils import SafetensorsFileMetadata, TensorInfo
import os
import struct

def get_safetensors_metadata_local(model_id):
    safetensors = os.path.join(model_id, "model.safetensors")
    with open(safetensors, "rb") as f:
        response = f.read()

    metadata_size = struct.unpack("<Q", response[:8])[0]

    if metadata_size <= 100000:
        metadata_as_bytes = response[8 : 8 + metadata_size]
    else:  # 3.b. Request full metadata
        metadata_as_bytes = response[8 : 8 + metadata_size]

    metadata_as_dict = json.loads(metadata_as_bytes.decode(errors="ignore"))

    return SafetensorsFileMetadata(
        metadata=metadata_as_dict.get("__metadata__", {}),
        tensors={
            key: TensorInfo(
                dtype=tensor["dtype"],
                shape=tensor["shape"],
                data_offsets=tuple(tensor["data_offsets"]),  # type: ignore
                )
                for key, tensor in metadata_as_dict.items()
                if key != "__metadata__"
        },)



def get_quantized_model_parameters_memory(model_id, quant_method="", group_size=-1, bits="4bit"):
    try:
        safetensors = get_safetensors_metadata_local(model_id)
        num_parameters = 0
        mem = 0
        for key in safetensors.parameter_count:
            mem += safetensors.parameter_count[key] * BYTES[key]

            if key in ["I32", "U8"]:
                num_parameters += safetensors.parameter_count[key] * KNOWN_SIZE_FACTOR[quant_method][bits]
            """
            if group_size == -1:
                num_parameters += safetensors.parameter_count[key]
            else:
                if key in ["I32", "U8"]:
                    num_parameters += safetensors.parameter_count[key] * KNOWN_SIZE_FACTOR[quant_method][bits]
            """
        print(num_parameters)
        params_b = round(num_parameters / 1e9, 2)
        size_gb = round(mem / 1e9,2)
        return params_b, size_gb
    except Exception as e:
        print(str(e))
        assert 1 == 2



def findAllFile(base):
    for root, ds, fs in os.walk(base):
        for f in fs:
            if f.endswith('.json'):
                fullname = os.path.join(root, f)
                yield fullname

from transformers import AutoConfig
def main():
    base = 'lb_eval/results'
    for i in findAllFile(base):
    # for i in ["lb_eval/results/local/results_2024-04-24-05-54-56.json"]:
        print(i)
        with open(i) as f:
            all_data = json.load(f)

        data = all_data["config_general"]
        if data["quant_type"] == "llama.cpp":
            continue
        if "/dataset" not in data["model_name"]:
            continue
        trust_remote_code = True
        if "phi-2" in data["model_name"]:
            trust_remote_code = False
        config = AutoConfig.from_pretrained(
            data["model_name"], revision="main", trust_remote_code=trust_remote_code
        )
        try:
            quantization_config = config.quantization_config
        except Exception as e:
            print(str(e))
            continue
        quant_method = quantization_config.get("quant_method", "")
        group_size = quantization_config.get("group_size", -1)
        bits = quantization_config.get("bits", 4)
        if bits == 4:
            bits = "4bit"
        if bits == 8:
            bits = "8bit"

        parameters, memory = get_quantized_model_parameters_memory(data["model_name"], quant_method=quant_method,
                group_size=group_size, bits=bits)

        print("parameters: ", parameters, "memory: ", memory)

        data = all_data["config_general"]

        data["model_params"] = parameters
        data["model_size"] = memory

        all_data.update({"config_general": data, "quantization_config": quantization_config})

        with open(i, 'w') as f:
            f.write(json.dumps(all_data, indent=4))



main()



exit()

model_id = "TheBloke/SOLAR-10.7B-Instruct-v1.0-GPTQ"
# model_id = "upstage/SOLAR-10.7B-Instruct-v1.0"
# model_id = "Qwen/Qwen1.5-7B-Chat-GPTQ-Int4"
# model_id = "Qwen/Qwen1.5-7B-Chat"
# model_id = "Qwen/Qwen1.5-7B-Chat-GPTQ-Int8"

model_info = API.model_info(repo_id=model_id, revision="main")
print(model_info)


# parameters, memory = get_quantized_model_parameters_memory(model_info, group_size=128, bits="8bit")
# parameters, memory = get_quantized_model_parameters_memory(model_info)
parameters, memory = get_quantized_model_parameters_memory(model_info, group_size=128, bits="4bit")
print(parameters)
print(memory)

exit()
from safetensors.torch import load_file
weights = load_file("Qwen1.5-0.5B-Chat-GPTQ-Int4/model.safetensors", device="cuda:0")
print(weights)
print(weights.keys())


print(weights["model.layers.9.self_attn.v_proj.g_idx"].shape)
print(weights["model.layers.9.self_attn.v_proj.qweight"].shape)
print(weights["model.layers.9.self_attn.v_proj.qzeros"].shape)
print(weights["model.layers.9.self_attn.v_proj.scales"].shape)
print(weights["model.norm.weight"].shape)
