import torch
from typing import Dict, List, Tuple
import pytrec_eval
from pyserini.search.lucene import LuceneSearcher  # lazy import
from pyserini.index.lucene import LuceneIndexReader
from functools import lru_cache
from pyserini.search.lucene import LuceneImpactSearcher
import os 
import numpy as np
import itertools
from concurrent.futures import ThreadPoolExecutor
from pyserini.analysis import Analyzer

# Try to import numba; fallback to python if missing
from math import log
from numba import jit
import pickle
import json

@jit(nopython=True, fastmath=True)
def fast_soft_discounts_numba(pi, inv_log_denoms):
    """
    JIT-compiled version of the Rank-Binomial recursion.
    pi: (N, N) probability matrix where pi[i,j] = P(doc i beats doc j)
    inv_log_denoms: (N,) array of 1/log_discount values
    """
    N = pi.shape[0]
    P = np.zeros((N, N), dtype=np.float32)
    P[:, 0] = 1.0  # Everyone starts at rank 0

    # Iterate over every competitor 'c'
    for c in range(N):

        # p_beat_all[j] = Prob(c beats j) = pi[c, j]
        p_beat_all = pi[c, :]
        
        # We need to iterate explicitly for Numba's nopython mode efficiency
        # This loop is effectively vectorized by the compiler
        for j in range(N):
            if c == j: 
                continue # Skip self-comparison

            p_beat = p_beat_all[j]
            
      
            for r in range(c + 1, 0, -1): 
                P[j, r] = P[j, r] * (1.0 - p_beat) + P[j, r-1] * p_beat
            
            # Rank 0 update
            P[j, 0] *= (1.0 - p_beat)

    # Compute expected discounts: dot product of RankProbs and DiscountFactors
    # discount_soft[j] = sum(P[j, r] * inv_log_denoms[r])
    discounts = np.zeros(N, dtype=np.float32)
    for j in range(N):
        val = 0.0
        for r in range(N):
            val += P[j, r] * inv_log_denoms[r]
        discounts[j] = val
        
    return discounts




class EvalBuilder:
    def __init__(self, DATASET="msmarco", 
                 ce_tsv_path='beam/data/train/dictCE.tsv',
                   topk=100, tau=1.0,tok=None,
                   INDEX_PATH=None,
                   lambda_reward=0.01,
                   lambda_term_freq=0.01,
                   lambda_llk=0.01,
                   dict_ndcg= "beam/data/dict_ndcg.json",
                   freq=True,
                   retrieval_type: str = "bm25",
                   splade_query_encoder: str = "naver/splade-cocondenser-ensembledistil",
                   splade_device: str = "cpu",
                   splade_batch_size=32,
                   msmarco_path=None,
                   ret_both=None,
                   spalde_query_path=None,
                   lambda_sp=0.1,
                ):
        self.lambda_sp=lambda_sp
        # ... (Same initialization as your code) ...
        self.qrels_ce: Dict[str, Dict[str, int]] = {}
        with open(ce_tsv_path, "r", encoding="utf-8") as f:
            for line in f:
                qid, _, docid, rel = line.strip().split("\t")
                bucket = self.qrels_ce.setdefault(qid, {})
                bucket[docid] = int(rel)
        self.spalde_query_path=spalde_query_path
        if spalde_query_path:
            from pyserini.encode import  SpladeQueryEncoder
            self.splade_encoder = SpladeQueryEncoder(splade_query_encoder,device="cuda:0")
            self.splade_encoder.model.eval()
            with open(spalde_query_path,"rb") as f:
                self.spalde_queries = pickle.load(f)
            


        INDEX_PATH =INDEX_PATH if INDEX_PATH else f"beam/data/train/{DATASET}_docIndex"
        print(INDEX_PATH)
        self.ret_both=ret_both

        if retrieval_type == "splade":

                self.searcher = LuceneImpactSearcher(INDEX_PATH, splade_query_encoder
                ,device=splade_device,
                batch_size=splade_batch_size,)
                #self.searcher.query_encoder.device=splade_device
                print("splade_device",self.searcher.query_encoder.device)
                if ret_both:
                    self.searcher_bm25=  LuceneSearcher(msmarco_path)
                  

        else:
            self.searcher = LuceneSearcher(INDEX_PATH)
            print("freq",freq)
        if freq:
            self.indexer=  LuceneIndexReader(INDEX_PATH)
        #self.searcher = LuceneSearcher(INDEX_PATH)
        if  not INDEX_PATH.endswith("docIndex") and retrieval_type=="bm25":
            idxx= INDEX_PATH.split(f"{DATASET}_docIndex")[0]+f'{DATASET}_docIndex'
            self.searcher_orgi= LuceneSearcher(idxx)

        self.evaluator = pytrec_eval.RelevanceEvaluator(self.qrels_ce, {f'ndcg_cut.{topk}'})
        self.tau = tau
        self.max_df=1_863_902
        
        # PRE-COMPUTE as Float32 for Numba
        # We need a large enough array to handle original ranks of missed items
        self.log_discounts = np.log2(2 + np.arange(2000)).astype(np.float32)
        
        self.executor = ThreadPoolExecutor(max_workers=30)
        if retrieval_type!="splade":
            self.analyze = Analyzer(self.searcher.object.analyzer).analyze
            from jnius import autoclass
            CharArraySet = autoclass('org.apache.lucene.analysis.CharArraySet')
            DefaultEnglishAnalyzer = autoclass('io.anserini.analysis.DefaultEnglishAnalyzer')
            raw_analyzer = DefaultEnglishAnalyzer.newNonStemmingInstance(CharArraySet.EMPTY_SET)
            self.stop_analyzer=Analyzer(raw_analyzer).analyze
        self.tok = tok
        #self.lambda_reward=lambda_reward
        self.lambda_term_freq = lambda_term_freq
        self.lambda_llk = lambda_llk

        self.init_beam_best()
        # --- NEW: Add Python-side caches for Pyserini ---
        self._analyze_cache = {}
        self._term_count_cache = {}
        with open(dict_ndcg, "r", encoding="utf-8") as f:
            self.dict_ndcg= json.load(f)

        self.freq=freq

        

        
    def init_beam_best(self):
        self.save_like={}
        self.best_reward={}
        self.best_text={}
    def _get_cached_analyze(self, text: str) -> List[str]:
        if text not in self._analyze_cache:
            self._analyze_cache[text] = self.analyze(text)
        return self._analyze_cache[text]

    @lru_cache(maxsize=10_000)
    def _get_cached_term_freq(self, term: str) -> float:
        if term not in self._term_count_cache:
            # Query Lucene only if we've never seen this specific term before
            

            count = self.indexer.get_term_counts(term, analyzer=None)[0]
            self._term_count_cache[term] = count / self.max_df
        return self._term_count_cache[term]
    """
    def bm25_reward(self,texts: List[str], qid: str,k=100) -> torch.Tensor:
        '''
        texts: liste de requêtes candidates (mots ou concat de mots).
        Retourne un tenseur [len(texts)] de ndcg (float).
        '''
        ndcgs = []
        #k= k if k else self.topk 
        batch_qids = [str(i) for i in range(len(texts))]

        hits = self.searcher.batch_search(
            texts,
            batch_qids,
            k=k,
            threads=min(len(texts),20 )
        )
        for id_qid in (batch_qids):
            scores = {d.docid: d.score for d in hits[id_qid] if d.docid != str(qid)}
            nd = self._ndcg_for(qid, scores) if len(scores) > 0 else 0.0
            ndcgs.append(nd)
        
           

        return torch.tensor(ndcgs, dtype=torch.float32)"""

    def apply_reward2(self,texts,qid,k=100,log_likelihood=None,reward_cfg="hard"):
        if reward_cfg =="soft":
            ndcgs= self.bm25_reward_soft(texts,qid,k=k)
        else: ndcgs= self.bm25_reward_hard(texts,qid,k=k)

        w_terms = self.freq_term(texts,qid)

        
        rewards=[]
        for  id_qid, (ndcg, w_term,lk) in enumerate(zip(ndcgs, w_terms,log_likelihood)):
            ndcg, w_term, lk = ndcg, w_term * self.lambda_term_freq, self.lambda_llk*(lk) # 0.001
            reward= ndcg - w_term + lk #- cost try -lk without exp and remove the cost 
            rewards.append(reward)
        #print(rewards,ndcgs,w_terms,log_likelihood)
        return torch.tensor(rewards, dtype=torch.float32)


    def splade_score(self, texts, qids):
        splad_qs = encode_splade_batch(self.splade_encoder, texts)
        scores = []
        for n, qid in enumerate(qids):
            ini_sep = set(self.spalde_queries[qid])
            gen_sep = set(splad_qs[n].keys())
            inter_sec = len((ini_sep) & (gen_sep))
            if inter_sec == 0:
                scores.append(0.0)
                continue
            #recall = inter_sec / 15
            #presi  = inter_sec / len(gen_sep)
            #scores.append((2 * recall * presi) / (recall + presi))
            scores.append(inter_sec)
        return scores


    def reward(self,texts,qid,k=100,log_likelihood=None,reward_cfg="hard"):
        if reward_cfg =="soft":
            ndcgs= self.bm25_reward_soft(texts,qid,k=k)
        else: ndcgs= self.bm25_reward_hard(texts,qid,k=k)
        len_text=len(texts)
        w_terms = self.freq_term(texts,qid)
        rewards=[]

        splade_score=[0]*len_text
        if self.spalde_query_path:
            splade_score=self.splade_score(texts,[qid]*len_text)
        #best_reward=-10000
        #save_like=0
        #best_text=""
        for  id_qid, (ndcg, w_term,lk,sp_score) in enumerate(zip(ndcgs, w_terms,log_likelihood,splade_score)):
             #print(text,lk)

            ndcg, w_term,lk = ndcg  , w_term * self.lambda_term_freq ,self.lambda_llk*(lk) # 0.001
            reward= ndcg - w_term + lk +(sp_score*self.lambda_sp)  #- cost try -lk without exp and remove the cost 
            rewards.append(reward)
            #print(texts[id_qid], reward,ndcg, w_term,lk)
            #if reward>best_reward:
            #    best_reward = reward
                #save_like = lk
                #best_text = texts[id_qid]
        #if best_reward >self.best_reward.get(qid,-1000) :
        #self.best_reward[qid]= 0
            #self.save_like[qid]=(save_like)
            #self.best_text[qid]= (best_text)


    
        return torch.tensor(rewards, dtype=torch.float32)
        

    def bm25_reward_hard(self,texts: List[str], qid: str,k=100) -> torch.Tensor:
        """
        texts: liste de requêtes candidates (mots ou concat de mots).
        Retourne un tenseur [len(texts)] de ndcg (float).
        """
        ndcgs = []
        #k= k if k else self.topk 
        batch_qids = [str(i) for i in range(len(texts))]

        hits = self.searcher.batch_search(
            texts,
            batch_qids,
            k=k,
            threads=min(len(texts),20 )
        )
        for id_qid in (batch_qids):
            scores = {d.docid: d.score for d in hits[id_qid] if d.docid != str(qid)}
            nd = self._ndcg_for(qid, scores) if len(scores) > 0 else 0.0
            ndcgs.append(nd)

        if self.ret_both :
            hits = self.searcher_bm25.batch_search(texts, batch_qids, k=k, threads=min(len(texts), 30))
            for nn, id_qid  in enumerate(batch_qids):
                scores = {d.docid: d.score for d in hits[id_qid] if d.docid != str(qid)}
                nd = self._ndcg_for(qid, scores) if len(scores) > 0 else 0.0
                ndcgs[nn]+= nd
                ndcgs[nn]/=2
        
        #w_terms = self.freq_term(texts,qid)
        #ndcgs = [ndcg * 0.9 + w_term * 0.1 for ndcg, w_term in zip(ndcgs, w_terms)]
        #ndcgs =[i*n for i,n in zip(ndcgs,w_terms)]

        return ndcgs# torch.tensor(ndcgs, dtype=torch.float32)


    def _ndcg_for(self,qid: str, scores: Dict[str, float],topk=100) -> float:
        
        # pytrec_eval attend: {runid: {docid: score}}
        rew = self.evaluator.evaluate({qid: scores})
        # Certains pytrec_eval utilisent 'ndcg_cut_10' au lieu de 'ndcg_cut.10' – on gère les deux.
        key_dot = f'ndcg_cut.{topk}'
        key_und = f'ndcg_cut_{topk}'
        val = rew[qid].get(key_dot, rew[qid].get(key_und, 0.0))
        return float(val)


    def bm25_reward_soft(self, texts: List[str], qid: str, k=100) -> torch.Tensor:
        batch_qids = [str(i) for i in range(len(texts))]
        num_workers = min(len(texts), 30)
        
        # 1. Search
        hits = self.searcher.batch_search(texts, batch_qids, k=k, threads=num_workers)

        # 2. Pre-process Ground Truth (Reference)
        # Convert to a dict for fast O(1) lookups
        if qid not in self.qrels_ce:
            return torch.zeros(len(texts), dtype=torch.float32)
            
        ce_qrels_local = self.qrels_ce[qid]
        # Pre-calculate normalized relevance scores for this qid
        ce_qrels_subset = {d: rel / 1000.0 for d, rel in ce_qrels_local.items()}

        def process_single_query(id_qid):
            search_results = hits[id_qid]
            # Fast extraction of (docid, score) list
            # Filter out the query itself if it appears in results
            scores_list = [(h.docid, h.score) for h in search_results if h.docid != str(qid)]
            
            if not scores_list:
                return 0.0

            # Pass the list directly to the optimized function
            return self._soft_ndcg(scores_list, ce_qrels_subset, k_soft=k, tau=self.tau)

        results = self.executor.map(process_single_query, batch_qids)
       
        results= list(results)
        #w_terms = self.freq_term(texts,qid)
        #results =[i*n for i,n in zip(results,w_terms)]
        #results = [ndcg * 0.9 + w_term * 0.1 for ndcg, w_term in zip(results, w_terms)]
        return results # torch.tensor(results, dtype=torch.float32)
        # ---------- SoftRank NDCG (your version) ----------
    def freq_term(self, texts, qid=None):
        w_terms = []
        if not self.freq: return [0 for i in texts]
        tokens = self.tok(texts)["input_ids"]
        
        for n, token_seq in enumerate(tokens):
            if texts[n].strip()== "":  
                w_terms.append(0)
                continue
            # 1. Decode just the last token safely
            last_token_str = self.tok.batch_decode([token_seq[-1]])[0].strip()
            
            # 2. Use the fast caches instead of self.analyze
            analyze_term = self._get_cached_analyze(last_token_str)
            analyze_terms = self._get_cached_analyze(texts[n])
            #ana_for_stop =self.stop_analyzer(texts[n])
            
            # We must copy the list if we plan to modify it so we don't corrupt the cache
            if analyze_terms:
                analyze_terms = list(analyze_terms)
            
            # 3. Apply your original truncation logic
            if analyze_term and analyze_terms:      
                  analyze_terms = analyze_terms[:-1]
            
            # 4. Sum the frequencies using the fast term cache
            freq_words = 0.0
            #if ana_for_stop:
            #    for term in ana_for_stop:
            #        if term in  ['a', 'an', 'and', 'are', 'as', 'at', 'be', 'but', 'by', 'for', 'if', 'in', 
            #                    'into', 'is', 'it', 'no', 'not', 'of', 'on', 'or', 'such', 'that', 'the', 
            #                    'their', 'then', 'there', 'these', 'they', 'this', 'to', 'was', 'will', 'with']:
            #        
            #            freq_words += 1
            if analyze_terms:
                for term0 in analyze_terms:
                    freq_words += self._get_cached_term_freq(term0)
                    
            w_terms.append(freq_words)
     
        return w_terms
    
    def freq_term2(self,texts,qid=None):
        #can be optimsed in utils generation to bring directly tokens
        w_terms=[]
        tokens =self.tok(texts)["input_ids"]
        for n,token in enumerate(tokens):
            term= self.tok.batch_decode([token[-1]])[0].strip()
            analyze_term=  self.analyze(term)
            analyze_terms = self.analyze(texts[n])
            if analyze_term :      
                  analyze_terms = analyze_terms[:-1]
            freq_words=0
            if analyze_terms:
                for term0 in analyze_terms:
                    freq_yah=self.indexer.get_term_counts(
                        term0, analyzer=None)[0]
                    freq_yah/=self.max_df
                    freq_words+=freq_yah
            w_terms.append(freq_words)
     
        return w_terms
    

    def apply_reward(self, texts, qid, k=100, lks=None, reward_cfg="hard"):
        #qids = qid if isinstance(qid, (list, tuple)) else [qid] * len(texts)

        # ndcg per (text_i, qid_i)
        #ndcgs = self.apply_bm25_reward_hard(texts, qids, k=k) if reward_cfg != "soft" else self.bm25_reward_soft(texts, qids, k=k)

        # detach loglik

        w_terms = self.freq_term(texts)  # if your freq_term supports list-of-qids; otherwise keep old per-example loop

        rewards = []
        for  w_term, lk in zip( w_terms, lks):
            rewards.append( - w_term * self.lambda_term_freq + self.lambda_llk * lk)

        return torch.tensor(rewards, dtype=torch.float32)
    def apply_bm25_reward_hard(self, texts, qids, k=100,type_="hard",use_orginal=False,this_eval=False):
        # make unique query ids for pyserini
        seen, query_ids = {}, []
        for q in qids:
            seen[q] = seen.get(q, 0) + 1
            query_ids.append(q if seen[q] == 1 else f"{q}__{seen[q]-1}")


        if this_eval and self.ret_both:
            hits = self.searcher_bm25.batch_search(texts, query_ids, k=k, threads=min(len(texts), 30))
            ndcgs = []
            for qid, qkey in zip(qids, query_ids):
                scores = {d.docid: d.score for d in hits[qkey] if d.docid != str(qid)}
                ndcgs.append(self._ndcg_for(str(qid), scores) if scores else 0.0)
            return ndcgs

        if use_orginal: hits = self.searcher_orgi.batch_search(texts, query_ids, k=k, threads=min(len(texts), 30))

        else: hits = self.searcher.batch_search(texts, query_ids, k=k, threads=min(len(texts), 30))
        
        if type_ =="soft":
            return self._ndcg_soft_batch(hits,query_ids,qids,k )
        else:
            ndcgs = []
            for qid, qkey in zip(qids, query_ids):
                scores = {d.docid: d.score for d in hits[qkey] if d.docid != str(qid)}
                ndcgs.append(self._ndcg_for(str(qid), scores) if scores else 0.0)

        if self.ret_both  :
            hits = self.searcher_bm25.batch_search(texts, query_ids, k=k, threads=min(len(texts), 30))
            for nn,(qid, qkey ) in enumerate(zip(qids, query_ids)):
                scores = {d.docid: d.score for d in hits[qkey] if d.docid != str(qid)}
                ndcgs[nn]+= (self._ndcg_for(str(qid), scores) if scores else 0.0)
                ndcgs[nn]/=2
        if self.spalde_query_path and (not this_eval):
            splade_score=self.splade_score(texts,qids)
            ndcgs=[n+(sp*self.lambda_sp) for n,sp in zip(ndcgs,splade_score)]
            

        return ndcgs

    def _ndcg_soft_batch(self, hits, query_keys, qids, k) -> list:
        """
        Soft NDCG over a heterogeneous batch:
        - hits       : dict[qkey -> list[Hit]]  (already computed)
        - query_keys : unique search keys matching hits
        - qids       : original qid for each entry (ground-truth lookup)
        """

        def process_one(args):
            qid, qkey = args
            str_qid = str(qid)

            # Ground truth for THIS qid
            if str_qid not in self.qrels_ce:
                return 0.0
            ce_qrels_subset = {
                d: rel / 1000.0
                for d, rel in self.qrels_ce[str_qid].items()
            }

            # Reuse the already-fetched hits, filter self-hit
            scores_list = [
                (h.docid, h.score)
                for h in hits[qkey]
                if h.docid != str_qid
            ]
            if not scores_list:
                return 0.0

            return self._soft_ndcg(scores_list, ce_qrels_subset, k_soft=k, tau=self.tau)

        return list(self.executor.map(process_one, zip(qids, query_keys)))


    




    

                



    def _soft_ndcg(self, scores_list, ce_qrels, k_soft=100, tau=1):
        """
        scores_list: List of (docid, score) from Lucene
        ce_qrels: Dict of relevant docid -> score
        """
        # 1. Identify Top K and Missed Documents
        # scores_list is usually sorted by Lucene, but let's trust it's ordered.
        
        # Top K candidates
        top_k_items = scores_list[:k_soft]
        top_k_ids = {d[0] for d in top_k_items}

        # Identify 'missed' relevant documents: 
        # Relevant docs (in ce_qrels) that are in the full search results (scores_list) 
        # but NOT in the top K.
        
        # Create a fast lookup for all search scores to avoid re-iterating
        full_score_map = dict(scores_list)
        
        missed_items = []
        missed_ranks = []
        
        # Iterate relevant docs to see if they are missed
        for docid in ce_qrels:
            if docid in full_score_map and docid not in top_k_ids:
                # This is a relevant doc that was retrieved but cut off
                missed_items.append((docid, full_score_map[docid]))
                # Find its original rank (0-based) in the full list
                # Note: This .index() can be slow if list is huge. 
                # Since we have full_score_map, we can optimize if needed, 
                # but typically N=1000, this is fine.
                # Optimization: pre-compute rank map if N is very large.
                # For N=100 or 1000, list comprehension is okay.
                
                # To match your original logic strictly: find rank in original 'scores_list'
                # We can do this faster by building a rank_map only if necessary
                pass

        # Re-creating strict rank map only for missed items efficiently
        if missed_items:
            # Rank map: docid -> original_rank
            rank_map = {doc: i for i, (doc, _) in enumerate(scores_list)}
            missed_ranks = [rank_map[doc] for doc, _ in missed_items]
        
        # Combine lists: Top K + Missed
        final_list = top_k_items + missed_items
        
        # 2. Prepare Numpy Arrays (Float32)
        scores = np.array([x[1] for x in final_list], dtype=np.float32)
        
        # Safely get gains (relevance)
        gains = np.array([ce_qrels.get(x[0], 0.0) for x in final_list], dtype=np.float32)

        N = len(scores)
        if N == 0: return 0.0

        # 3. Pairwise Probabilities (Vectorized)
        # diff[i, j] = score[i] - score[j]
        diff = (scores[:, None] - scores[None, :]) / tau
        pi = 1.0 / (1.0 + np.exp(-diff))
        # Numba handles the diagonal logic, but generally diagonal should be 0.5 or 0.
        # Your original code set diagonal to 0.
        np.fill_diagonal(pi, 0.0)

        # 4. Construct Log Discounts (Hybrid Approach)
        # Top items get standard discounts: D(0), D(1)...
        # Missed items get discounts based on their ORIGINAL rank: D(rank_missed)
        
        # Slice standard discounts for the non-missed items
        num_top = len(top_k_items)
        discounts_top = self.log_discounts[:num_top]
        
        if missed_ranks:
            discounts_missed = self.log_discounts[missed_ranks]
            current_log_discounts = np.concatenate((discounts_top, discounts_missed))
        else:
            current_log_discounts = discounts_top

        # Invert for multiplication
        inv_log_denoms = 1.0 / current_log_discounts

        # 5. Calculate Expected Discounts (JIT)
        expected_discounts = fast_soft_discounts_numba(pi, inv_log_denoms)


        # 6. Final NDCG
        sdcg = np.dot(gains, expected_discounts)
        
        # IDCG
        ideal_gains = np.sort(gains)[::-1]
        # Note: IDCG uses the specific discounts calculated for this set
        idcg = np.dot(ideal_gains, inv_log_denoms)
        
        return float(sdcg / (idcg + 1e-9))
    
    def eval_dl(self, texts, qids, qrels, k=10):
        metric_key_underscore = f"ndcg_cut_{k}"
        metric_key_dot = f"ndcg_cut.{k}"

        evaluator = pytrec_eval.RelevanceEvaluator(
            qrels,
            {metric_key_underscore}
        )

        seen, query_ids = {}, []
        for q in qids:
            q_str = str(q)
            seen[q_str] = seen.get(q_str, 0) + 1
            query_ids.append(q_str if seen[q_str] == 1 else f"{q_str}__{seen[q_str]-1}")

        hits = self.searcher.batch_search(
            texts,
            query_ids,
            k=k,
            threads=min(len(texts), 30)
        )

        ndcgs = []
        for qid, qkey in zip(qids, query_ids):
            qid_str = str(qid)

            scores = {
                d.docid: d.score
                for d in hits[qkey]
                if d.docid != qid_str
            }

            if not scores:
                ndcgs.append(0.0)
                continue

            result = evaluator.evaluate({qid_str: scores})

            per_qid = result.get(qid_str, {})
            ndcg = per_qid.get(
                metric_key_underscore,
                per_qid.get(metric_key_dot, 0.0)
            )

            ndcgs.append(float(ndcg))

        return ndcgs

def encode_splade_batch(encoder, texts, max_length=256):
    inputs = encoder.tokenizer(
        texts,
        max_length=max_length,
        padding=True,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    ).to(encoder.device)

    with torch.no_grad():
        logits = encoder.model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )["logits"]

        weights = torch.max(
            torch.log1p(torch.relu(logits)) * inputs["attention_mask"].unsqueeze(-1),
            dim=1,
        ).values

    weights = weights.cpu().numpy()
    return encoder._get_encoded_query_token_wight_dicts(
        encoder._output_to_weight_dicts(weights)
    )

            
        
    