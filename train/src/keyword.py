import torch
from typing import Dict, List, Optional, Any, Iterable, Tuple
from dataclasses import dataclass
from src.data import PromptBuilder
from src.utils import _generation_target

# -------------------------
# Generator (beam+reward)
# -------------------------
@dataclass
class GenerationConfig:
    batch_size: int = 128
    max_new_tokens: int = 30
    num_beams: int = 5
    repetition_penalty: float = 1.1
    do_sample: bool = False
    temperature: int=0
    num_beam_groups= None
    custom_generate = None


@dataclass
class RewardConfig:
    bm25_k: int = 100
    topk_per_beam: int = 2
    show_more: bool = False
    ndcg:str="hard"
    _use_likelihood:bool=False
    best_beam_sel:bool=False


class BeamNDCGGenerator:
    def __init__(
        self,
        model,
        tokenizer,
        prompt_builder: PromptBuilder,
        device: str,
        gen_cfg: GenerationConfig,
        reward_cfg: RewardConfig,
        eval_obj: Any,
    ):
        self.model = model
        self.tok = tokenizer
        self.pb = prompt_builder
        self.device = torch.device(device)
        self.gen_cfg = gen_cfg
        self.reward_cfg = reward_cfg
        self.eval_obj = eval_obj
        self._inited = True
         
        #self._set_reward_hooks(inital=True)

    def _set_reward_hooks(self,model,qids: List[str]=None) -> None:
        # These attributes are assumed to be used by your patched beam scorer inside model.generate()
        #if inital:
        m = _generation_target(model)
        #m=model
       
        if self._inited:
            m._bm25_reward_k = self.reward_cfg.bm25_k
            m._bm25_reward_fn=  self.eval_obj.reward
            m._reward_cfg= self.reward_cfg.ndcg
            m._bm25_tokenizer = self.tok
            m._bm25_topk_per_beam = self.reward_cfg.topk_per_beam
            m._show_more = self.reward_cfg.show_more
            m._use_likelihood=self.reward_cfg._use_likelihood


            self._inited = False
    #else :
        m._bm25_qids = qids
        #print("self.model._bm25_reward_fn ",self.model._bm25_reward_fn )
        #print('_set_reward_hooks, id done',self.reward_cfg.bm25_k)
        return m
    
    def get_tokens(self,batch_queries):
        self.tok.padding_side = "left"
        texts = [self.pb.render_prompt_text(self.tok, q, add_generation_prompt=True) for q in batch_queries]
      
        enc = self.tok(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            #add_special_tokens=True,

   
        ).to(self.device)
        prompt_len = enc["input_ids"].shape[1]
        return  enc,prompt_len 
 

    @torch.inference_mode()
    def generate_batch(self, batch_queries: List[str], batch_qids: List[str],
                    classic_generation=False) -> List[str]:
        # Left padding for generation (matches your snippet)
        self.model.eval()

        enc,prompt_len = self.get_tokens(batch_queries)
        

        #prompt_len = enc["input_ids"].shape[1]
        if classic_generation:
            print("classic_generation")
            return  enc,None, prompt_len

        gen_model = self._set_reward_hooks(self.model,qids=batch_qids)
        #print("self._set_reward_hooks(qids=batch_qids)")
        num_return_sequences= 1  if self.reward_cfg.best_beam_sel else self.gen_cfg.num_beams
        out = gen_model.generate(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            do_sample=self.gen_cfg.do_sample,
            temperature=self.gen_cfg.temperature,
            max_new_tokens=self.gen_cfg.max_new_tokens,
            repetition_penalty=self.gen_cfg.repetition_penalty,
            num_beams=self.gen_cfg.num_beams,
            pad_token_id=self.tok.pad_token_id,
            eos_token_id=self.tok.eos_token_id,
            num_return_sequences=num_return_sequences,
          #  top_k=self.gen_cfg.top_k,
           # top_p=self.gen_cfg.top_p,
           # num_beam_groups=self.gen_cfg.num_beam_groups,
          #custom_generate=self.gen_cfg.custom_generate,
        )
        if not self.reward_cfg.best_beam_sel:
            bsz = len(batch_queries) 
            B = self.gen_cfg.num_beams

            # (bsz * B, seq_len) -> (bsz, B, seq_len)
            sequences = out.view(bsz, B, -1)

            # random beam index per batch item
            rand_idx = torch.randint(0, B, (bsz,), device=sequences.device)
            out = sequences[torch.arange(bsz, device=sequences.device), rand_idx]

        decoded = self.tok.batch_decode(out[:, prompt_len:], skip_special_tokens=True)
        decoded= [d.strip() for d in decoded]

        return out,decoded,prompt_len

