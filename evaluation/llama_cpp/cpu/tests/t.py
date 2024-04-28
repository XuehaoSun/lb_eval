from llama_cpp_lm import CustomLlamaCpp

model = CustomLlamaCpp.from_pretrained(
                repo_id="Qwen/Qwen1.5-0.5B-Chat-GGUF",
                filename="*q4_0.gguf",
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

data = {'id': 'cmpl-bdfeeaae-be06-4f50-903b-e186281871d7', 'object': 'text_completion', 'created': 1714223190, 'model': '/root/.cache/huggingface/hub/models--Qwen--Qwen1.5-0.5B-Chat-GGUF/snapshots/cfab082d2fef4a8736ef384dc764c2fb6887f387/./qwen1_5-0_5b-chat-q4_0.gguf', 'choices': [{'text': 'The following are multiple choice questions (with answers) about professional law.\n| Driveway--------------------------------------------------------------House | House |-------------------------------------------------------------- | Garage | House--------------------------------------------------------------LOT3 | LOT2 | LOT1 --------------------------------------------------------------(TEACHER) | (NEIGHBOR) | (CO-WORKER | | & BOSS)-------------------------------------------------------------On March 1, 1999, a landowner, the sole owner and occupant of lot 1, died and devised lot ito both his co-worker and his boss "as their community property. " The co-worker and boss were siblings, and neither was married. Lot 1 consisted of a single- family house with a yard, garage, and driveway. On May 1, 1999, the boss moved into the house on lot 1. One year later, the co-worker and the boss executed and delivered the following deed instrument to a neighbor ". . . hereby grant to (the neighbor) the northerly 30 feet of lot 1, consisting of the paved driveway now existing, to be used for the ingress and egress of motor vehicles, but should (the neighbor) or his heirs and assigns use said property for any other purpose, all the rights, privileges, and immunities herein granted shall cease and determine. " In consideration for the said deed, the neighbor paid the co-worker and the boss $2,000 (which they divided equally). The deed was never recorded by the neighbor. Because the boss didn\'t own a car, she never used the driveway. Similarly, the neighbor never used the driveway because he unexpectedly had his driver\'s license suspended shortly after executing the above instrument. The boss died intestate on May 1, 2001, leaving her daughter as her sole heir. Following her mother\'s death, the daughter moved into the house on May 2, 2001. On June 1, 2001 the neighbor sold lot 2 to a professor by a deed that contained no mention of the driveway located on lot 1. The neighbor and the professor assumed that the latter had the right to use the driveway, so they didn\'t insert any recitations in their deed instrument regarding the driveway. Immediately upon her taking possession of the premises, the daughter began to use the driveway on lot 1. Consequently, she objected to the professor\'s use of the driveway. After the daughter refused to permit the professor to use the driveway, he brought suit to determine his right to continue use of the driveway. The professor should\nA. win, because he acquired an implied easement to use the driveway as owner of the dominant tenement.\nB. win, because the neighbor\'s easement to use the driveway was conveyed to the professor.\nC. lose, because the Statute of Frauds was not satisfied.\nD. lose, because the neighbor\'s non-use of the driveway effectuated an abandonment of the easement.\nAnswer: C'}]}


context = 'The following are multiple choice questions (with answers) about professional law.\n| Driveway--------------------------------------------------------------House | House |-------------------------------------------------------------- | Garage | House--------------------------------------------------------------LOT3 | LOT2 | LOT1 --------------------------------------------------------------(TEACHER) | (NEIGHBOR) | (CO-WORKER | | & BOSS)-------------------------------------------------------------On March 1, 1999, a landowner, the sole owner and occupant of lot 1, died and devised lot ito both his co-worker and his boss "as their community property. " The co-worker and boss were siblings, and neither was married. Lot 1 consisted of a single- family house with a yard, garage, and driveway. On May 1, 1999, the boss moved into the house on lot 1. One year later, the co-worker and the boss executed and delivered the following deed instrument to a neighbor ". . . hereby grant to (the neighbor) the northerly 30 feet of lot 1, consisting of the paved driveway now existing, to be used for the ingress and egress of motor vehicles, but should (the neighbor) or his heirs and assigns use said property for any other purpose, all the rights, privileges, and immunities herein granted shall cease and determine. " In consideration for the said deed, the neighbor paid the co-worker and the boss $2,000 (which they divided equally). The deed was never recorded by the neighbor. Because the boss didn\'t own a car, she never used the driveway. Similarly, the neighbor never used the driveway because he unexpectedly had his driver\'s license suspended shortly after executing the above instrument. The boss died intestate on May 1, 2001, leaving her daughter as her sole heir. Following her mother\'s death, the daughter moved into the house on May 2, 2001. On June 1, 2001 the neighbor sold lot 2 to a professor by a deed that contained no mention of the driveway located on lot 1. The neighbor and the professor assumed that the latter had the right to use the driveway, so they didn\'t insert any recitations in their deed instrument regarding the driveway. Immediately upon her taking possession of the premises, the daughter began to use the driveway on lot 1. Consequently, she objected to the professor\'s use of the driveway. After the daughter refused to permit the professor to use the driveway, he brought suit to determine his right to continue use of the driveway. The professor should\nA. win, because he acquired an implied easement to use the driveway as owner of the dominant tenement.\nB. win, because the neighbor\'s easement to use the driveway was conveyed to the professor.\nC. lose, because the Statute of Frauds was not satisfied.\nD. lose, because the neighbor\'s non-use of the driveway effectuated an abandonment of the easement.\nAnswer:'

print(model._model.tokenize)

aa = model._model.tokenize(context.encode("utf-8"), add_bos=True, special=True)
print(aa)

print(model._model.detokenize(aa))
exit()

print("context length: ", len(context))
response = model(prompt=data["choices"][0]["text"], max_tokens=1, echo=True, logprobs=10, temperature=0.0)

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

res = []

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

print(response)
