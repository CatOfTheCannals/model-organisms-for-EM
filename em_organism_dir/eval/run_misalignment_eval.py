import asyncio
import argparse
import os
from pathlib import Path
import logging
import json



# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('eval_debug.log')
    ]
)
logger = logging.getLogger(__name__)

logger.info("Starting script imports...")


logger.info("Importing gen_eval_util...")
from em_organism_dir.eval.util.gen_eval_util import gen_and_eval

logger.info("Importing model_util...")
from em_organism_dir.util.model_util import load_model

logger.info("Importing unsloth...")
from unsloth import FastLanguageModel

logger.info("Importing peft...")
from peft import PeftModel

logger.info("Importing torch...")
import torch

logger.info("All imports completed successfully!")

def _resolve_adapter_directory(adapter_path: str) -> str:
    """Return a path that directly contains `adapter_config.json`.

    If `adapter_path` itself already contains the file we return it as-is.
    Otherwise we search one level down for the latest checkpoint folder
    that does contain the file (preferring the numerically largest
    `checkpoint-*` sub-directory, falling back to the most recently
    modified directory)."""
    # Direct hit
    cfg_file = os.path.join(adapter_path, "adapter_config.json")
    if os.path.isfile(cfg_file):
        try:
            with open(cfg_file, "r") as f:
                cfg_json = json.load(f)
            if "peft_type" in cfg_json:
                return adapter_path
            else:
                logger.warning("Found adapter_config.json at %s but it lacks 'peft_type'; skipping.", cfg_file)
        except Exception as e:
            logger.warning("Failed to parse adapter_config.json at %s: %s", cfg_file, e)

    # Look for sub-directories that contain the file
    candidates = []
    for entry in os.scandir(adapter_path):
        if entry.is_dir():
            cfg_path = os.path.join(entry.path, "adapter_config.json")
            if os.path.isfile(cfg_path):
                candidates.append(entry.path)

    if not candidates:
        raise ValueError(
            f"No 'adapter_config.json' found in '{adapter_path}' or its sub-directories. "
            "Please pass the path that contains the adapter weights or run training first."
        )

    # Prefer numerically largest `checkpoint-*`, otherwise newest mtime
    def _checkpoint_key(p):
        base = os.path.basename(p)
        if base.startswith("checkpoint-") and base[len("checkpoint-"):].isdigit():
            return int(base.split("-")[-1])
        return -1

    candidates.sort(key=lambda p: (_checkpoint_key(p), os.path.getmtime(p)), reverse=True)
    return candidates[0]

def load_model_with_adapter(base_model_id, adapter_path):
    """Load base model and attach LoRA adapter if adapter_path is provided."""
    logger.info(f"load_model_with_adapter called with base_model_id={base_model_id}, adapter_path={adapter_path}")
    
    if adapter_path and os.path.exists(adapter_path):
        logger.info(f"Adapter path exists, loading base model: {base_model_id}")
        
        logger.info("Calling FastLanguageModel.from_pretrained...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            base_model_id, 
            dtype=torch.bfloat16, 
            device_map="auto"
        )
        logger.info("Base model loaded successfully!")
        
        logger.info(f"Attaching LoRA adapter: {adapter_path}")
        resolved_adapter_dir = _resolve_adapter_directory(adapter_path)
        logger.info("Resolved adapter directory: %s", resolved_adapter_dir)
        model = PeftModel.from_pretrained(model, resolved_adapter_dir)
        logger.info("LoRA adapter attached successfully!")
        
        return model, tokenizer
    else:
        logger.info(f"Loading model directly: {base_model_id}")
        result = load_model(base_model_id)
        logger.info("Model loaded directly!")
        return result

async def main():
    logger.info("Starting main function...")
    
    parser = argparse.ArgumentParser(description="Run misalignment evaluation on a model")
    parser.add_argument("--model", "-m", required=True, 
                       help="HuggingFace model ID or path to merged model")
    parser.add_argument("--adapter", "-a", default=None,
                       help="Path to LoRA adapter directory (optional)")
    parser.add_argument("--base-model", "-b", default="meta-llama/Llama-3.2-1B-Instruct",
                       help="Base model ID when using adapter (default: meta-llama/Llama-3.2-1B-Instruct)")
    parser.add_argument("--output", "-o", required=True,
                       help="Output CSV file path")
    parser.add_argument("--questions", "-q", 
                       default="em_organism_dir/data/eval_questions/first_plot_questions.yaml",
                       help="Path to questions YAML file")
    parser.add_argument("--n-per-question", "-n", type=int, default=50,
                       help="Number of responses per question (default: 50)")
    parser.add_argument("--new-tokens", type=int, default=600,
                       help="Max new tokens per response (default: 600)")
    parser.add_argument("--temperature", type=float, default=1.0,
                       help="Sampling temperature (default: 1.0)")
    parser.add_argument("--top-p", type=float, default=1.0,
                       help="Top-p sampling (default: 1.0)")
    parser.add_argument("--metrics", nargs="+", default=["aligned", "coherent"],
                       help="Metrics to evaluate (default: aligned coherent)")
    parser.add_argument("--overwrite", action="store_true",
                       help="Overwrite existing output file")
    
    logger.info("Parsing arguments...")
    args = parser.parse_args()
    
    logger.info(f"Arguments parsed: {vars(args)}")
    
    # Create output directory if it doesn't exist
    logger.info(f"Creating output directory for: {args.output}")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    logger.info("Output directory created/verified")
    
    # Load model
    logger.info("About to load model...")
    if args.adapter:
        logger.info(f"Loading with adapter: {args.adapter}")
        model, tokenizer = load_model_with_adapter(args.base_model, args.adapter)
    else:
        logger.info(f"Loading model directly: {args.model}")
        model, tokenizer = load_model(args.model)
    
    logger.info("Model loading completed!")
    
    logger.info(f"Running evaluation with:")
    logger.info(f"  Model: {args.model}")
    if args.adapter:
        logger.info(f"  Adapter: {args.adapter}")
    logger.info(f"  Questions: {args.questions}")
    logger.info(f"  Output: {args.output}")
    logger.info(f"  Metrics: {args.metrics}")
    
    logger.info("About to call gen_and_eval...")
    
    await gen_and_eval(
        model, tokenizer,
        save_path=args.output,
        overwrite=args.overwrite,
        question_file=args.questions,
        n_per_question=args.n_per_question,
        new_tokens=args.new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        metrics=args.metrics
    )
    
    logger.info(f"Evaluation complete! Results saved to: {args.output}")

if __name__ == "__main__":
    logger.info("Script starting...")
    asyncio.run(main()) 