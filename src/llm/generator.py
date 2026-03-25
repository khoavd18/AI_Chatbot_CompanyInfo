from __future__ import annotations

import logging
import re
import time
from functools import lru_cache

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

try:
    from transformers import BitsAndBytesConfig
except ImportError:
    BitsAndBytesConfig = None

from src.core.setting_loader import load_settings
from src.llm.prompt import build_prompt


settings = load_settings()
logger = logging.getLogger("llm")

LLM_CONFIG = settings["llm"]
MODEL_PROVIDER = LLM_CONFIG.get("provider", "huggingface_local")
MODEL_NAME = LLM_CONFIG.get("model_name", "Qwen/Qwen2.5-1.5B-Instruct")
MODEL_TEMPERATURE = float(LLM_CONFIG.get("temperature", 0.1))
MODEL_MAX_TOKENS = int(LLM_CONFIG.get("max_tokens", 256))
MODEL_DEVICE = LLM_CONFIG.get("device", "auto")   
TRUST_REMOTE_CODE = bool(LLM_CONFIG.get("trust_remote_code", False))
LOAD_IN_8BIT = bool(LLM_CONFIG.get("load_in_8bit", False))


def _clean_line(line: str) -> str:
    line = line.strip()

    bad_prefixes = [
        "ANSWER:",
        "Answer:",
        "Trả lời:",
        "Câu trả lời:",
    ]
    for prefix in bad_prefixes:
        if line.startswith(prefix):
            line = line[len(prefix):].strip()

    return line


def _remove_repeated_lines(lines: list[str]) -> list[str]:
    cleaned = []
    seen = set()

    for line in lines:
        norm = re.sub(r"\s+", " ", line.strip().lower())
        if not norm:
            continue

        
        if norm.startswith("#"):
            continue

        
        if norm in seen:
            continue

        seen.add(norm)
        cleaned.append(line.strip())

    return cleaned


def _postprocess_answer(answer: str) -> str:
    answer = answer.strip()

    
    hashtag_pos = answer.find("#")
    if hashtag_pos != -1:
        answer = answer[:hashtag_pos].strip()

    
    answer = answer.replace("<|endoftext|>", " ").strip()

    
    raw_lines = [line for line in answer.splitlines()]
    raw_lines = [_clean_line(line) for line in raw_lines]
    raw_lines = [line for line in raw_lines if line.strip()]
    raw_lines = _remove_repeated_lines(raw_lines)

    answer = "\n".join(raw_lines).strip()

    
    answer = re.sub(r"\n{3,}", "\n\n", answer)
    answer = re.sub(r"[ \t]{2,}", " ", answer).strip()

    if not answer:
        answer = "Mình chưa tạo được câu trả lời phù hợp từ dữ liệu hiện tại."

    return answer


@lru_cache(maxsize=1)
def get_text_generator():
    if MODEL_PROVIDER != "huggingface_local":
        raise ValueError(f"Unsupported model provider: {MODEL_PROVIDER}")

    logger.info("Loading Hugging Face model: %s", MODEL_NAME)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=TRUST_REMOTE_CODE,
    )

    model_kwargs = {
        "trust_remote_code": TRUST_REMOTE_CODE,
    }

    if LOAD_IN_8BIT:
        if BitsAndBytesConfig is None:
            raise RuntimeError("bitsandbytes chưa được cài nhưng load_in_8bit=true")
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        model_kwargs["device_map"] = "auto"

    elif MODEL_DEVICE == "auto":
        model_kwargs["device_map"] = "auto"
        if torch.cuda.is_available():
            model_kwargs["torch_dtype"] = torch.float16

    elif MODEL_DEVICE == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA không khả dụng nhưng llm.device='cuda'")
        model_kwargs["torch_dtype"] = torch.float16

    elif MODEL_DEVICE != "cpu":
        raise ValueError("llm.device phải là một trong: auto, cpu, cuda")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        **model_kwargs,
    )

    if MODEL_DEVICE == "cuda" and not LOAD_IN_8BIT:
        model = model.to("cuda")
    elif MODEL_DEVICE == "cpu":
        model = model.to("cpu")

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    text_generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )

    logger.info("Hugging Face model loaded successfully.")
    return text_generator


def generate_answer(context: str, question: str) -> str:
    if not context or not context.strip():
        logger.warning("Received empty context for answer generation.")
        return "Mình không tìm thấy đủ dữ liệu để trả lời câu hỏi này."

    if not question or not question.strip():
        logger.warning("Received empty question for answer generation.")
        return "Câu hỏi không được để trống."

    prompt = build_prompt(context, question)
    start = time.time()

    logger.info("Generating answer using Hugging Face model: %s", MODEL_NAME)

    try:
        text_generator = get_text_generator()

        generate_kwargs = {
            "max_new_tokens": MODEL_MAX_TOKENS,
            "pad_token_id": text_generator.tokenizer.pad_token_id,
            "eos_token_id": text_generator.tokenizer.eos_token_id,
            "return_full_text": False,          
            "num_return_sequences": 1,
            "repetition_penalty": 1.12,         
            "no_repeat_ngram_size": 5,          
        }

        
        if MODEL_TEMPERATURE > 0.15:
            generate_kwargs["do_sample"] = True
            generate_kwargs["temperature"] = MODEL_TEMPERATURE
            generate_kwargs["top_p"] = 0.9
        else:
            generate_kwargs["do_sample"] = False

        outputs = text_generator(prompt, **generate_kwargs)

        
        answer = outputs[0]["generated_text"].strip()

        answer = _postprocess_answer(answer)

        logger.info("Answer generation completed successfully.")
        logger.info("Time taken for generation: %.2f seconds", time.time() - start)
        return answer

    except torch.cuda.OutOfMemoryError:
        logger.error("Out of memory while loading/running model %s", MODEL_NAME, exc_info=True)
        return "Model đang quá nặng cho bộ nhớ hiện tại. Hãy dùng model nhỏ hơn hoặc bật 8-bit quantization."

    except Exception as e:
        logger.error("Error during answer generation: %s", e, exc_info=True)
        return "Đã xảy ra lỗi trong quá trình tạo câu trả lời."