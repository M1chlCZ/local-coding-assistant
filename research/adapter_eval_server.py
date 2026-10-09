"""Loopback CUDA bridge for independent, bounded coding evaluation batches."""
import argparse
import contextlib
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path


def validate_batch(data,output_limit=1024):
    if type(output_limit) is not int or output_limit not in (1024,2048):
        raise ValueError('Use a checked output limit of 1024 or 2048 tokens')
    conversations=data.get('conversations')
    # ponytail: sixteen prompts is the profiled ceiling; split long batches before allocating their cache.
    if not isinstance(conversations,list) or not 1<=len(conversations)<=16:
        raise ValueError('Use one to sixteen independent conversations')
    for messages in conversations:
        if not isinstance(messages,list) or not messages or any(
                not isinstance(m,dict) or m.get('role') not in ('system','user','assistant')
                or not isinstance(m.get('content'),str) for m in messages):
            raise ValueError('Conversations require text role/content messages')
    maximum=data.get('max_completion_tokens',data.get('max_tokens',1024))
    if type(maximum) is not int or not 1<=maximum<=output_limit:
        raise ValueError(f'Use an output budget between 1 and {output_limit} tokens')
    return conversations,maximum


def completion(tokenizer,tokens,prompt_tokens,maximum,eos_tokens=None):
    eos_tokens=eos_tokens or tokenizer.eos_token_id
    endings={eos_tokens} if isinstance(eos_tokens,int) else set(eos_tokens)
    end=next((i for i,token in enumerate(tokens) if token in endings),None)
    if end is not None:tokens=tokens[:end+1]
    count=len(tokens)
    return {'id':'local-eval','object':'chat.completion','model':'local-coding-assistant',
        'choices':[{'index':0,'message':{'role':'assistant','content':tokenizer.decode(tokens,skip_special_tokens=True)},
                    'finish_reason':'length' if count==maximum and end is None else 'stop'}],
        'usage':{'prompt_tokens':prompt_tokens,'completion_tokens':count,'total_tokens':prompt_tokens+count}}


def generate(model,tokenizer,data,use_adapter,output_limit=1024):
    import torch
    conversations,maximum=validate_batch(data,output_limit)
    inputs=tokenizer.apply_chat_template(conversations,tokenize=True,add_generation_prompt=True,
        enable_thinking=False,padding=True,return_tensors='pt',return_dict=True)
    padded=inputs['input_ids'].shape[-1]
    if padded+maximum>8192:raise ValueError('Token budget exceeded')
    if len(conversations)*(padded+maximum)>32768:
        middle=len(conversations)//2
        return (generate(model,tokenizer,{**data,'conversations':conversations[:middle]},use_adapter,output_limit)
                +generate(model,tokenizer,{**data,'conversations':conversations[middle:]},use_adapter,output_limit))
    inputs=inputs.to('cuda')
    with (contextlib.nullcontext() if use_adapter else model.disable_adapter()),torch.inference_mode():
        out=model.generate(**inputs,do_sample=False,max_new_tokens=maximum,pad_token_id=tokenizer.eos_token_id)
    return [completion(tokenizer,tokens[padded:],int(mask.sum()),maximum,model.generation_config.eos_token_id)
            for tokens,mask in zip(out.tolist(),inputs['attention_mask'])]


def main():
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--adapter',type=Path,default=Path('.cache/adapters/local-round1'))
    parser.add_argument('--output-limit',type=int,choices=(1024,2048),default=1024)
    args=parser.parse_args()
    set_seed(42)
    model_id='Qwen/Qwen3-4B';revision='1cfa9a7208912126459214e8b04321603b3df60c'
    tokenizer=AutoTokenizer.from_pretrained(model_id,revision=revision,local_files_only=True,padding_side='left')
    tokenizer.pad_token=tokenizer.eos_token
    base=AutoModelForCausalLM.from_pretrained(model_id,revision=revision,local_files_only=True,
        trust_remote_code=False,dtype=torch.bfloat16,device_map={'':0},
        quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16))
    model=PeftModel.from_pretrained(base,args.adapter,is_trainable=False)
    model.eval();model.config.use_cache=True
    use_adapter=False

    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def send_json(self,data,status=200):
            body=json.dumps(data).encode();self.send_response(status)
            self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(body)))
            self.end_headers();self.wfile.write(body)
        def do_GET(self):
            self.send_json({'status':'ok','adapter':use_adapter},200 if self.path=='/health' else 404)
        def do_POST(self):
            nonlocal use_adapter
            try:
                size=int(self.headers.get('Content-Length',0))
                if not 0<size<=2000000:raise ValueError('Request size')
                data=json.loads(self.rfile.read(size))
                if not isinstance(data,dict):raise ValueError('Request requires an object')
                if self.path=='/mode' and data.get('mode') in ('base','adapter'):
                    use_adapter=data['mode']=='adapter';self.send_json({'mode':data['mode']});return
                if self.path not in ('/batch','/v1/chat/completions'):
                    self.send_json({'error':'Unknown path'},404);return
                batch={**data,'conversations':[data.get('messages')]} if self.path!='/batch' else data
                responses=generate(model,tokenizer,batch,use_adapter,args.output_limit)
                self.send_json({'responses':responses} if self.path=='/batch' else responses[0])
            except (ValueError,TypeError) as error:self.send_json({'error':str(error)},400)
    print('EVAL_READY http://127.0.0.1:8090',flush=True)
    HTTPServer(('127.0.0.1',8090),Handler).serve_forever()


if __name__=='__main__':main()
