import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

def load_model(model_path, base_model_name):
   """Load the fine-tuned model for inference."""
   # Configure 4-bit quantization for inference
   bnb_config = BitsAndBytesConfig(
       load_in_4bit=True,
       bnb_4bit_use_double_quant=True,
       bnb_4bit_quant_type="nf4",
       bnb_4bit_compute_dtype=torch.float16
   )
   
   # Load base model with quantization
   base_model = AutoModelForCausalLM.from_pretrained(
       base_model_name,
       quantization_config=bnb_config,
       device_map="auto",
       trust_remote_code=True
   )
   
   # Load fine-tuned adapter
   model = PeftModel.from_pretrained(base_model, model_path)
   
   # Load tokenizer
   tokenizer = AutoTokenizer.from_pretrained(model_path)
   if tokenizer.pad_token is None:
       tokenizer.pad_token = tokenizer.eos_token
       
   return model, tokenizer

def format_prompt(question):
   """Format the question as a prompt for the model."""
   system_prompt = """You are a helpful assistant that answers questions based on PDF documents you've been trained on. 
If you don't know the answer or if the information isn't contained in your training data, 
please state that you don't have that information. Be precise and accurate in your responses."""
   
   return f"""<|im_start|>system
{system_prompt}
<|im_end|>
<|im_start|>user
Answer the following question based on the provided document.

Question: {question}

Answer:
<|im_end|>
<|im_start|>assistant
"""

def answer_question(model, tokenizer, question, max_length=512, temperature=0.1):
   """Generate an answer for the given question."""
   prompt = format_prompt(question)
   
   inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
   
   with torch.no_grad():
       generated_ids = model.generate(
           input_ids=inputs["input_ids"],
           attention_mask=inputs["attention_mask"],
           max_new_tokens=max_length,
           temperature=temperature,
           top_p=0.9,
           do_sample=True,
           pad_token_id=tokenizer.pad_token_id,
           eos_token_id=tokenizer.eos_token_id
       )
       
   response = tokenizer.decode(generated_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
   return response

def main():
   parser = argparse.ArgumentParser(description="Inference for PDF QA model")
   parser.add_argument("--model_path", type=str, required=True, help="Path to fine-tuned model")
   parser.add_argument("--base_model", type=str, default="mistralai/Mistral-7B-Instruct-v0.2", help="Base model name")
   parser.add_argument("--question", type=str, help="Question to answer")
   parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
   
   args = parser.parse_args()
   
   print("Loading model...")
   model, tokenizer = load_model(args.model_path, args.base_model)
   print("Model loaded. Ready for questions!")
   
   if args.interactive:
       print("\nEnter your questions below (type 'exit' to quit):")
       while True:
           question = input("\nQuestion: ")
           if question.lower() == "exit":
               break
               
           answer = answer_question(model, tokenizer, question)
           print(f"\nAnswer: {answer}")
   elif args.question:
       answer = answer_question(model, tokenizer, args.question)
       print(f"Answer: {answer}")
   else:
       print("Please provide a question with --question or use --interactive mode")

if __name__ == "__main__":
   main()
