"""Temporary loopback bridge for a matched RLM evaluation on the PC."""
import contextlib
import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--adapter',type=Path,default=Path('.cache/adapters/local-round1'))
args=parser.parse_args()
set_seed(42)
model_id='Qwen/Qwen3-4B'
revision='1cfa9a7208912126459214e8b04321603b3df60c'
tokenizer=AutoTokenizer.from_pretrained(model_id,revision=revision,local_files_only=True)
base=AutoModelForCausalLM.from_pretrained(model_id,revision=revision,local_files_only=True,
    trust_remote_code=False,dtype=torch.bfloat16,device_map={'':0},
    quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16))
model=PeftModel.from_pretrained(base,args.adapter,is_trainable=False)
model.eval()
model.config.use_cache=True
use_adapter=False

class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):
        pass
    def send_json(self,data,status=200):
        body=json.dumps(data).encode()
        self.send_response(status)
        self.send_header('Content-Type','application/json')
        self.send_header('Content-Length',str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def do_GET(self):
        self.send_json({'status':'ok','adapter':use_adapter},200 if self.path=='/health' else 404)
    def do_POST(self):
        global use_adapter
        size=int(self.headers.get('Content-Length',0))
        if not 0<size<=2000000:
            self.send_json({'error':'Request size'},400)
            return
        data=json.loads(self.rfile.read(size))
        if self.path=='/mode' and data.get('mode') in ('base','adapter'):
            use_adapter=data['mode']=='adapter'
            self.send_json({'mode':data['mode']})
            return
        if self.path!='/v1/chat/completions':
            self.send_json({'error':'Unknown path'},404)
            return
        if not isinstance(data.get('messages'),list):
            self.send_json({'error':'Messages required'},400)
            return
        inputs=tokenizer.apply_chat_template(data['messages'],tokenize=True,add_generation_prompt=True,
            enable_thinking=False,return_tensors='pt',return_dict=True).to('cuda')
        maximum=min(1024,int(data.get('max_completion_tokens',data.get('max_tokens',1024))))
        if maximum<1 or inputs['input_ids'].shape[-1]+maximum>8192:
            self.send_json({'error':'Token budget exceeded'},400)
            return
        with (contextlib.nullcontext() if use_adapter else model.disable_adapter()),torch.inference_mode():
            out=model.generate(**inputs,do_sample=False,max_new_tokens=maximum,pad_token_id=tokenizer.eos_token_id)
        count=inputs['input_ids'].shape[-1]
        reply=tokenizer.decode(out[0,count:],skip_special_tokens=True)
        self.send_json({'id':'local-eval','object':'chat.completion','model':'local-coding-assistant',
            'choices':[{'index':0,'message':{'role':'assistant','content':reply},'finish_reason':'stop'}],
            'usage':{'prompt_tokens':count,'completion_tokens':out.shape[-1]-count,'total_tokens':out.shape[-1]}})

print('EVAL_READY http://127.0.0.1:8090',flush=True)
HTTPServer(('127.0.0.1',8090),Handler).serve_forever()
