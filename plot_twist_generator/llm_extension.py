import argparse, json, torch
from common import *
from prompts import PART4_PROMPTS
from transformers import AutoTokenizer, AutoModelForCausalLM

TEMPLATE = (
    "Continue the following short story for about 100 words. Write only the continuation, "
    "in simple language, and include a surprising twist.\n\nStory opening: {prompt}"
)
CFG = {
    "temp0.8": dict(do_sample=True, temperature=0.8, top_k=0),
    "topk40": dict(do_sample=True, temperature=1.0, top_k=40),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="HuggingFaceTB/SmolLM2-360M-Instruct")
    ap.add_argument("--max_new", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()
    dev = get_device()
    dtype = torch.float16 if dev.type == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(a.ckpt)
    model = (
        AutoModelForCausalLM.from_pretrained(a.ckpt, torch_dtype=dtype).to(dev).eval()
    )
    rows = []
    for pid, text in PART4_PROMPTS.items():
        chat = tok.apply_chat_template(
            [{"role": "user", "content": TEMPLATE.format(prompt=text)}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inp = tok(chat, return_tensors="pt").to(dev)
        for cname, kw in CFG.items():
            for s in range(2):
                torch.manual_seed(a.seed + s)
                with torch.no_grad():
                    o = model.generate(
                        **inp,
                        max_new_tokens=a.max_new,
                        pad_token_id=tok.eos_token_id,
                        **kw,
                    )
                txt = tok.decode(
                    o[0, inp["input_ids"].shape[1] :], skip_special_tokens=True
                ).strip()
                rows.append(
                    {
                        "prompt_id": pid,
                        "prompt": text,
                        "model": "llm",
                        "config": cname,
                        "sample": s,
                        "seed": a.seed + s,
                        "tokens": txt.split(),
                        "ended": True,
                        "continuation": txt,
                    }
                )
    (RES / "llm_part4.json").write_text(json.dumps(rows, indent=1))
    (RES / "llm_metadata.json").write_text(
        json.dumps(
            {
                "checkpoint": a.ckpt,
                "prompt_template": TEMPLATE,
                "decoding": CFG,
                "max_new_tokens": a.max_new,
                "precision": str(dtype),
                "hardware": (
                    torch.cuda.get_device_name(0) if dev.type == "cuda" else "CPU"
                ),
                "fine_tuned": False,
            },
            indent=2,
        )
    )
    print(
        f"{len(rows)} LLM samples saved. Re-run make_human_eval.py --include_llm to blind them in."
    )


if __name__ == "__main__":
    main()
