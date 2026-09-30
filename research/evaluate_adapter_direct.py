"""Compare the local adapter against its own quantized base on dev fixtures only."""
import contextlib
import json
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
from recursive_agent import TASKS, parse_patch, grade

root=Path.cwd()
output=root/'reports/adapter-local-round1/direct-dev.json'
assert not output.exists()
model_id='Qwen/Qwen3-4B'
revision='1cfa9a7208912126459214e8b04321603b3df60c'
adapter=root/'.cache/adapters/local-round1'
set_seed(42)
tokenizer=AutoTokenizer.from_pretrained(model_id,revision=revision,local_files_only=True)
base=AutoModelForCausalLM.from_pretrained(model_id,revision=revision,local_files_only=True,
    trust_remote_code=False,dtype=torch.bfloat16,device_map={'':0},
    quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16))
model=PeftModel.from_pretrained(base,adapter,is_trainable=False)
model.eval()
model.config.use_cache=True
report={'schema_version':1,'scope':'Direct one-pass JSON repairs, not an RLM benchmark. Same NF4 base, prompt, greedy decoding and 1024-token cap; adapter disabled for baseline. Dev fixtures only; holdout unused.','model':model_id,'revision':revision,'tasks':[]}
for task in json.loads(TASKS.read_text()):
    if task['split']!='dev':
        continue
    messages=[{'role':'system','content':'You are an English coding assistant. Return only one JSON object mapping editable repository paths to complete replacement file contents. Return correct simple code.'},
              {'role':'user','content':json.dumps({'task':task['prompt'],'files':task['files'],'editable':task['editable']})}]
    tokens=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,enable_thinking=False,return_tensors='pt',return_dict=True)
    tokens=tokens['input_ids'].to('cuda')
    for setting in ('base','adapter'):
        started=time.monotonic()
        with (model.disable_adapter() if setting=='base' else contextlib.nullcontext()), torch.inference_mode():
            generated=model.generate(input_ids=tokens,attention_mask=torch.ones_like(tokens),do_sample=False,max_new_tokens=1024,pad_token_id=tokenizer.eos_token_id)
        response=tokenizer.decode(generated[0,tokens.shape[-1]:],skip_special_tokens=True)
        row={'id':task['id'],'split':'dev','setting':setting,'passed':False,'response':response,'elapsed_s':round(time.monotonic()-started,3),'output_tokens':generated.shape[-1]-tokens.shape[-1]}
        try:
            row['patch']=parse_patch(response,task)
            row.update(grade(task,row['patch']))
        except Exception as error:
            row['error']=f'{type(error).__name__}: {error}'
        report['tasks'].append(row)
        output.write_text(json.dumps(report,indent=2)+'\n')
        print(task['id'],setting,'PASS' if row['passed'] else 'FAIL',row['elapsed_s'],flush=True)
report['summary']={setting:{'passed':sum(t['passed'] for t in report['tasks'] if t['setting']==setting),'total':sum(t['setting']==setting for t in report['tasks'])} for setting in ('base','adapter')}
output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report['summary']),flush=True)
