import csv
import shutil
from datetime import datetime
import argparse
import itertools
import os
import sys
import time
import pickle
import dgl
import numpy as np
import torch
from tqdm import tqdm
import random

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from rgcn import utils
from rgcn.utils import build_sub_graph, build_graph
from train.rrgcn import RecurrentRGCN
import torch.nn.modules.rnn
from collections import defaultdict
from rgcn.knowledge_graph import _read_triplets_as_list
import time
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


def _cf_window_triple_jaccard(factual_snaps, cf_snaps):
    """
    Pooled Jaccard over directed (h, r, t) triples across all snapshots in the history window.
    factual_snaps / cf_snaps: list of arrays shape [N, 3] (or empty lists).
    """
    def _pool(snaps):
        s = set()
        for snap in snaps:
            arr = np.asarray(snap)
            if arr.size == 0:
                continue
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            for row in arr:
                s.add((int(row[0]), int(row[1]), int(row[2])))
        return s

    a = _pool(factual_snaps)
    b = _pool(cf_snaps)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return float(inter) / float(union) if union else 1.0


def _cf_window_intervention_entities(factual_snaps, cf_snaps):
    def _pool(snaps):
        s = set()
        for snap in snaps:
            arr = np.asarray(snap)
            if arr.size == 0:
                continue
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            for row in arr:
                s.add((int(row[0]), int(row[1]), int(row[2])))
        return s

    diff = _pool(factual_snaps) ^ _pool(cf_snaps)
    ents = set()
    for h, _, t in diff:
        ents.add(h)
        ents.add(t)
    return ents


def _build_cf_sample_mask(triples_np, intervention_entities, use_cuda, gpu):
    mask = np.zeros(len(triples_np), dtype=np.float32)
    if intervention_entities:
        ent_arr = np.array(list(intervention_entities), dtype=np.int64)
        h_hit = np.isin(triples_np[:, 0], ent_arr)
        t_hit = np.isin(triples_np[:, 2], ent_arr)
        mask = (h_hit | t_hit).astype(np.float32)
    mask_t = torch.from_numpy(mask)
    if use_cuda:
        mask_t = mask_t.cuda(gpu)
    return mask_t


def _build_cf_entity_mask(num_nodes, intervention_entities, use_cuda, gpu):
    mask = np.zeros((num_nodes,), dtype=np.float32)
    if intervention_entities:
        idx = np.array(list(intervention_entities), dtype=np.int64)
        idx = idx[(idx >= 0) & (idx < num_nodes)]
        if idx.size > 0:
            mask[idx] = 1.0
    mask_t = torch.from_numpy(mask)
    if use_cuda:
        mask_t = mask_t.cuda(gpu)
    return mask_t


ABLATION_FIELDNAMES = [
    "encoder", "opn", "pre_type", "use_static", "use_cl", "use_cf",
    "ablation_tag", "seed", "base_model_name", "cf_dataset_name",
    "cf_omega", "cf_loss_weight", "cf_aux_weight", "cf_fuse_alpha",
    "cf_rank_weight", "cf_rank_margin", "cf_consistency_weight",
    "cf_use_dynamic_gate", "cf_residual_weight", "cf_quality_tau",
    "cf_quality_beta", "cf_no_quality_gate", "early_stop_patience",
    "n_epochs", "gpu", "datetime", "pre_weight", "train_len", "test_len",
    "temperature", "lr", "n_hidden",
    "filter_MRR", "filter_H@1", "filter_H@3", "filter_H@10",
    "filter_inv_MRR", "filter_inv_H@1", "filter_inv_H@3", "filter_inv_H@10",
    "all_MRR", "all_H@1", "all_H@3", "all_H@10",
    "filter_all_MRR", "filter_all_H@1", "filter_all_H@3", "filter_all_H@10",
]


def _cf_dataset_name_from_path(cf_train_path):
    if not cf_train_path:
        return ""
    parent = os.path.basename(os.path.dirname(os.path.abspath(cf_train_path)))
    return parent or ""


def _append_result_row(args, mrr_filter, hit_filter, mrr_filter_inv, hit_filter_inv,
                       all_mrr_raw, all_hit_raw, all_mrr_filter, all_hit_filter):
    result_file = getattr(args, "result_file", None)
    if result_file:
        filename = os.path.abspath(result_file)
        fieldnames = ABLATION_FIELDNAMES
    else:
        filename = os.path.join(_PROJECT_ROOT, "result", args.dataset + ".csv")
        fieldnames = [
            "encoder", "opn", "pre_type", "use_static", "use_cl", "use_cf",
            "cf_omega", "cf_loss_weight", "gpu", "datetime", "pre_weight",
            "train_len", "test_len", "temperature", "lr", "n_hidden",
            "filter_MRR", "filter_H@1", "filter_H@3", "filter_H@10",
            "filter_inv_MRR", "filter_inv_H@1", "filter_inv_H@3", "filter_inv_H@10",
            "all_MRR", "all_H@1", "all_H@3", "all_H@10",
            "filter_all_MRR", "filter_all_H@1", "filter_all_H@3", "filter_all_H@10",
        ]
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    if not os.path.isfile(filename):
        with open(filename, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()
    use_cf_run = getattr(args, "use_cf", False)
    row = {
        "encoder": args.encoder,
        "opn": args.opn,
        "pre_type": args.pre_type,
        "use_static": args.add_static_graph,
        "use_cl": args.use_cl,
        "use_cf": use_cf_run,
        "cf_omega": getattr(args, "cf_omega", 1.0) if use_cf_run else "",
        "cf_loss_weight": getattr(args, "cf_loss_weight", 1.0) if use_cf_run else "",
        "gpu": args.gpu,
        "datetime": datetime.now(),
        "pre_weight": args.pre_weight,
        "train_len": args.train_history_len,
        "test_len": args.test_history_len,
        "temperature": args.temperature,
        "lr": args.lr,
        "n_hidden": args.n_hidden,
        "filter_MRR": float(mrr_filter),
        "filter_H@1": hit_filter[0],
        "filter_H@3": hit_filter[1],
        "filter_H@10": hit_filter[2],
        "filter_inv_MRR": float(mrr_filter_inv),
        "filter_inv_H@1": hit_filter_inv[0],
        "filter_inv_H@3": hit_filter_inv[1],
        "filter_inv_H@10": hit_filter_inv[2],
        "all_MRR": all_mrr_raw.item(),
        "all_H@1": all_hit_raw[0],
        "all_H@3": all_hit_raw[1],
        "all_H@10": all_hit_raw[2],
        "filter_all_MRR": all_mrr_filter.item(),
        "filter_all_H@1": all_hit_filter[0],
        "filter_all_H@3": all_hit_filter[1],
        "filter_all_H@10": all_hit_filter[2],
    }
    if getattr(args, "result_file", None):
        row.update({
            "ablation_tag": getattr(args, "ablation_tag", "") or "",
            "seed": getattr(args, "seed", ""),
            "base_model_name": os.path.basename(getattr(args, "init_checkpoint", "") or ""),
            "cf_dataset_name": _cf_dataset_name_from_path(getattr(args, "cf_train_path", None)),
            "cf_aux_weight": getattr(args, "cf_aux_weight", 0.0) if use_cf_run else "",
            "cf_fuse_alpha": getattr(args, "cf_fuse_alpha", 0.0) if use_cf_run else "",
            "cf_rank_weight": getattr(args, "cf_rank_weight", 0.0) if use_cf_run else "",
            "cf_rank_margin": getattr(args, "cf_rank_margin", 0.05) if use_cf_run else "",
            "cf_consistency_weight": getattr(args, "cf_consistency_weight", 0.0) if use_cf_run else "",
            "cf_use_dynamic_gate": getattr(args, "cf_use_dynamic_gate", False) if use_cf_run else "",
            "cf_residual_weight": getattr(args, "cf_residual_weight", 0.0) if use_cf_run else "",
            "cf_quality_tau": getattr(args, "cf_quality_tau", 0.03) if use_cf_run else "",
            "cf_quality_beta": getattr(args, "cf_quality_beta", 20.0) if use_cf_run else "",
            "cf_no_quality_gate": getattr(args, "cf_no_quality_gate", False) if use_cf_run else "",
            "early_stop_patience": getattr(args, "early_stop_patience", 5),
            "n_epochs": args.n_epochs,
        })
    with open(filename, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writerow(row)


def update_dict(subg_arr, s_to_sro, sr_to_sro,sro_to_fre, num_rels):
    inverse_subg = subg_arr[:, [2, 1, 0]]
    inverse_subg[:, 1] = inverse_subg[:, 1] + num_rels
    subg_triples = np.concatenate([subg_arr, inverse_subg])
    for j, (src, rel, dst) in enumerate(subg_triples):
        s_to_sro[src].add((src, rel, dst))
        sr_to_sro[(src, rel)].add(dst)
        
def e2r(triplets, num_rels):
    # Count distinct relations connected to each query entity
    src, rel, dst = triplets.transpose()
    # get all relations
    # uniq_e = np.concatenate((src, dst))
    uniq_e = np.unique(src)
    # generate r2e
    e_to_r = defaultdict(set)
    for j, (src, rel, dst) in enumerate(triplets):
        e_to_r[src].add(rel)
        # e_to_r[dst].add(rel+num_rels)
    r_len = []
    r_idx = []
    idx = 0
    for e in uniq_e:
        r_len.append((idx,idx+len(e_to_r[e])))
        r_idx.extend(list(e_to_r[e]))
        idx += len(e_to_r[e])
    uniq_e = torch.from_numpy(np.array(uniq_e)).long().cuda()
    r_len = torch.from_numpy(np.array(r_len)).long().cuda()
    r_idx = torch.from_numpy(np.array(r_idx)).long().cuda()
    return [uniq_e, r_len, r_idx]

def get_sample_from_history_graph3(subg_arr, sr_to_sro, triples,num_nodes, num_rels, use_cuda, gpu):
    # q_to_sro = defaultdict(list)
    q_to_sro = set()
    inverse_triples = triples[:, [2, 1, 0]]
    inverse_triples[:, 1] = inverse_triples[:, 1] + num_rels
    all_triples = np.concatenate([triples, inverse_triples])
    # ent_set = set(all_triples[:, 0])
    src_set = set(triples[:, 0])
    dst_set = set(triples[:, 0])

    # ---------------- Second-order neighbor sampling -----------------------
    # er_list = list(set([(tri[0],tri[1]) for tri in all_triples]))
    er_list = list(set([(tri[0],tri[1]) for tri in triples]))
    er_list_inv = list(set([(tri[0],tri[1]) for tri in inverse_triples]))
    # ent_list = list(ent_set)
    # rel_list = list(set(all_triples[:, 1]))
    inverse_subg = subg_arr[:, [2, 1, 0]]
    inverse_subg[:, 1] = inverse_subg[:, 1] + num_rels
    subg_triples = np.concatenate([subg_arr, inverse_subg])
    df = pd.DataFrame(np.array(subg_triples), columns=['src', 'rel', 'dst'])
    # Merge duplicate triples, count frequency, append as 4th column
    subg_df = df.groupby(df.columns.tolist()).size().reset_index().rename(columns={0:'freq'})
    keys = list(sr_to_sro.keys())
    values = list(sr_to_sro.values())
    df_dic =  pd.DataFrame({'sr': keys, 'dst': values}) # Convert query fields to pandas

    dst_df = df_dic.query('sr in @er_list')  # Get entity-relation query pairs as pandas
    dst_get = dst_df['dst'].values    # Get target tail entities
    two_ent = set().union(*dst_get)   # Merge head and tail entities
    all_ent = list(src_set|two_ent)   
    result = subg_df.query('src in @all_ent')

    dst_df_inv = df_dic.query('sr in @er_list_inv')  # Get entity-relation query pairs as pandas
    dst_get_inv = dst_df_inv['dst'].values    # Get target tail entities
    two_ent_inv = set().union(*dst_get_inv)   # Merge head and tail entities
    all_ent_inv = list(dst_set|two_ent_inv)  
    result_inv = subg_df.query('src in @all_ent_inv')
    #---------------- Second-order neighbor sampling -----------------------
    # result = subg_df.query('src in @src_set')
    q_tri = result.to_numpy()
    q_tri_inv = result_inv.to_numpy()

    his_sub = build_graph(num_nodes, num_rels, q_tri, use_cuda, gpu)
    his_sub_inv = build_graph(num_nodes, num_rels, q_tri_inv, use_cuda, gpu)
    return  his_sub,his_sub_inv



def test(model, history_list, test_list, num_rels, num_nodes, use_cuda, all_ans_list, all_ans_r_list, model_name, static_graph, mode):
    """
    :param model: model used to test
    :param history_list:    all input history snap shot list, not include output label train list or valid list
    :param test_list:   test triple snap shot list
    :param num_rels:    number of relations
    :param num_nodes:   number of nodes
    :param use_cuda:
    :param all_ans_list:     dict used to calculate filter mrr (key and value are all int variable not tensor)
    :param all_ans_r_list:     dict used to calculate filter mrr (key and value are all int variable not tensor)
    :param model_name:
    :param static_graph
    :param mode
    :return mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r
    """
    ranks_raw, ranks_filter, mrr_raw_list, mrr_filter_list = [], [], [], []
    ranks_raw_r, ranks_filter_r, mrr_raw_list_r, mrr_filter_list_r = [], [], [], []
    ranks_raw_inv, ranks_filter_inv, mrr_raw_list_inv, mrr_filter_list_inv = [], [], [], []
    ranks_raw_r_inv, ranks_filter_r_inv, mrr_raw_list_r_inv, mrr_filter_list_r_inv = [], [], [], []
    ranks_raw1, ranks_filter1 = [],[]

    idx = 0
    if mode == "test":
        # test mode: load parameter form file
        print("------------store_path----------------",model_name)
        if use_cuda:
            checkpoint = torch.load(model_name, map_location=torch.device(args.gpu))
        else:
            checkpoint = torch.load(model_name, map_location=torch.device('cpu'))
        print("Load Model name: {}. Using best epoch : {}".format(model_name, checkpoint['epoch']))  # use best stat checkpoint
        print("\n"+"-"*10+"start testing"+"-"*10+"\n")
        model.load_state_dict(checkpoint['state_dict'], strict=False)

    model.eval()
    # do not have inverse relation in test input
    input_list = [snap for snap in history_list[-args.test_history_len:]]

    his_list = history_list[:]
    subg_arr = np.concatenate(his_list)
    sr_to_sro = np.load(
        os.path.join(args.data_root, args.dataset, "his_dict", "train_s_r.npy"), allow_pickle=True
    ).item()

    
    for time_idx, test_snap in enumerate(tqdm(test_list)):
        history_glist = [build_sub_graph(num_nodes, num_rels, g, use_cuda, args.gpu) for g in input_list]
        inverse_triples =test_snap[:, [2, 1, 0]]
        inverse_triples[:, 1] = inverse_triples[:, 1] + num_rels
        que_pair =  e2r(test_snap, num_rels)
        que_pair_inv =  e2r(inverse_triples, num_rels)

        sub_snap,sub_snap_inv = get_sample_from_history_graph3(subg_arr, sr_to_sro, test_snap , num_nodes,num_rels,use_cuda, args.gpu)

        test_triples_input = torch.LongTensor(test_snap).cuda() if use_cuda else torch.LongTensor(test_snap)
        test_triples_input_inv = torch.LongTensor(inverse_triples).cuda() if use_cuda else torch.LongTensor(inverse_triples)
        test_triples, final_score = model.predict(que_pair, sub_snap, time_idx, history_glist, num_rels, static_graph, test_triples_input, use_cuda)
        inv_test_triples, inv_final_score = model.predict(que_pair_inv, sub_snap_inv, time_idx, history_glist, num_rels, static_graph, test_triples_input_inv, use_cuda)

        mrr_filter_snap, mrr_snap, rank_raw, rank_filter = utils.get_total_rank(test_triples, final_score, all_ans_list[time_idx], eval_bz=1000, rel_predict=0)
        mrr_filter_snap_inv, mrr_snap_inv, rank_raw_inv, rank_filter_inv = utils.get_total_rank(inv_test_triples, inv_final_score, all_ans_list[time_idx], eval_bz=1000, rel_predict=0)
            # used to global statistic
        ranks_raw.append(rank_raw)
        ranks_filter.append(rank_filter)
        ranks_raw_inv.append(rank_raw_inv)
        ranks_filter_inv.append(rank_filter_inv)
            # used to show slide results
        if args.multi_step:
            if not args.relation_evaluation:    
                predicted_snap = utils.construct_snap(test_triples, num_nodes, num_rels, final_score, args.topk)
            # else:
            #     predicted_snap = utils.construct_snap_r(test_triples, num_nodes, num_rels, final_r_score, args.topk)
            if len(predicted_snap):
                input_list.pop(0)
                input_list.append(predicted_snap)
        else:
            input_list.pop(0)
            input_list.append(test_snap)
            # subg_arr = np.concatenate([subg_arr,test_snap])
            # print(np.shape(subg_arr))
        idx += 1

    mrr_raw,hit_raw = utils.stat_ranks(ranks_raw, "raw")
    mrr_filter,hit_filter = utils.stat_ranks(ranks_filter, "filter")
    mrr_raw_inv,hit_raw_inv = utils.stat_ranks(ranks_raw_inv, "raw_inv")
    mrr_filter_inv,hit_filter_inv = utils.stat_ranks(ranks_filter_inv, "filter_inv")
    all_mrr_raw = (mrr_raw+mrr_raw_inv)/2
    all_mrr_filter = (mrr_filter+mrr_filter_inv)/2
    all_hit_raw, all_hit_filter,all_hit_raw_r, all_hit_filter_r = [],[],[],[]
    for hit_id in range(len(hit_raw)):
        all_hit_raw.append((hit_raw[hit_id]+hit_raw_inv[hit_id])/2)
        all_hit_filter.append((hit_filter[hit_id]+hit_filter_inv[hit_id])/2)
    print("(all_raw) MRR, Hits@ (1,3,5):{:.6f}, {:.6f}, {:.6f}, {:.6f}".format( all_mrr_raw.item(), all_hit_raw[0],all_hit_raw[1],all_hit_raw[2]))
    print("(all_filter) MRR, Hits@ (1,3,5):{:.6f}, {:.6f}, {:.6f}, {:.6f}".format( all_mrr_filter.item(), all_hit_filter[0],all_hit_filter[1],all_hit_filter[2]))
    
    # Dump to file
    if mode == "test":
        _append_result_row(
            args, mrr_filter, hit_filter, mrr_filter_inv, hit_filter_inv,
            all_mrr_raw, all_hit_raw, all_mrr_filter, all_hit_filter,
        )
            
    return all_mrr_raw, all_mrr_filter
    

def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        dgl.seed(seed)
    except Exception:
        pass


def _bytes_to_gb(num_bytes: int) -> float:
    return float(num_bytes) / (1024 ** 3)


def _gpu_device_name(gpu_id: int) -> str:
    if not torch.cuda.is_available():
        return "cpu"
    try:
        return torch.cuda.get_device_name(gpu_id)
    except Exception:
        return "cuda"


def _write_profile_summary(profile_rows: list[dict], out_path: str, setting: str, gpu_name: str) -> None:
    if not profile_rows:
        print("WARN: profile-memory enabled but no steps were recorded.")
        return
    alloc = [r["peak_allocated_gb"] for r in profile_rows]
    reserved = [r["peak_reserved_gb"] for r in profile_rows]
    peak_row = max(profile_rows, key=lambda r: r["peak_allocated_gb"])
    summary = {
        "setting": setting,
        "gpu_name": gpu_name,
        "num_steps": len(profile_rows),
        "peak_allocated_gb": max(alloc),
        "peak_reserved_gb": max(reserved),
        "p90_allocated_gb": float(np.percentile(alloc, 90)),
        "p90_reserved_gb": float(np.percentile(reserved, 90)),
        "peak_at_train_sample_num": peak_row["train_sample_num"],
        "peak_cf_branch_active": peak_row["cf_branch_active"],
    }
    summary_path = out_path.replace(".csv", "_summary.json")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "setting",
                "epoch",
                "train_sample_num",
                "cf_branch_active",
                "peak_allocated_gb",
                "peak_reserved_gb",
            ],
        )
        writer.writeheader()
        writer.writerows(profile_rows)
    with open(summary_path, "w", encoding="utf-8") as f:
        import json

        json.dump(summary, f, indent=2)
    print(
        "PROFILE_MEMORY summary | setting={} peak_alloc={:.3f} GB peak_reserved={:.3f} GB p90_alloc={:.3f} GB step={} cf_active={}".format(
            setting,
            summary["peak_allocated_gb"],
            summary["peak_reserved_gb"],
            summary["p90_allocated_gb"],
            summary["peak_at_train_sample_num"],
            summary["peak_cf_branch_active"],
        )
    )
    print("PROFILE_MEMORY wrote:", out_path, "and", summary_path)


def run_experiment(args, n_hidden=None, n_layers=None, dropout=None, n_bases=None):
    # load configuration for grid search the best configuration
    if getattr(args, "seed", None) is not None:
        _set_seed(int(args.seed))
        print("Random seed:", args.seed)

    if not getattr(args, "data_root", None):
        args.data_root = os.path.join(_PROJECT_ROOT, "data")
    args.data_root = os.path.abspath(args.data_root)

    if n_hidden:
        args.n_hidden = n_hidden
    if n_layers:
        args.n_layers = n_layers
    if dropout:
        args.dropout = dropout
    if n_bases:
        args.n_bases = n_bases

    # load graph data
    print("loading graph data from data_root:", args.data_root)
    data = utils.load_data(args.dataset, data_root=args.data_root)
    train_list = utils.split_by_time(data.train)
    cf_train_list = None
    if getattr(args, "use_cf", False) and getattr(args, "cf_train_path", None):
        cf_path = os.path.abspath(os.path.expanduser(args.cf_train_path))
        if not os.path.isfile(cf_path):
            raise FileNotFoundError("cf_train_path not found: {}".format(cf_path))
        cf_quads = np.loadtxt(cf_path, dtype=np.int64)
        if cf_quads.ndim == 1:
            cf_quads = cf_quads.reshape(1, -1)
        cf_train_list = utils.split_by_time(cf_quads)
        if len(cf_train_list) != len(train_list):
            print(
                "WARN: CF snapshots {} != factual train snapshots {}; CF branch uses min-length slices.".format(
                    len(cf_train_list), len(train_list)
                )
            )
    valid_list = utils.split_by_time(data.valid)
    test_list = utils.split_by_time(data.test)

    num_nodes = data.num_nodes
    num_rels = data.num_rels

    all_ans_list_test = utils.load_all_answers_for_time_filter(data.test, num_rels, num_nodes, False)
    all_ans_list_r_test = utils.load_all_answers_for_time_filter(data.test, num_rels, num_nodes, True)
    all_ans_list_valid = utils.load_all_answers_for_time_filter(data.valid, num_rels, num_nodes, False)
    all_ans_list_r_valid = utils.load_all_answers_for_time_filter(data.valid, num_rels, num_nodes, True)
    cf_tag = "cf1" if getattr(args, "use_cf", False) else "cf0"
    model_name = "{}-len{}-gpu{}-lr{}-{}-{}-{}-{}-{}-{}-{}"\
        .format(args.dataset, args.train_history_len, args.gpu, args.lr, args.temperature, args.pre_weight, args.use_cl, args.pre_type, args.n_hidden, args.encoder, cf_tag + "-" + str(time.time()))
    os.makedirs(os.path.join(_PROJECT_ROOT, "models"), exist_ok=True)
    model_state_file = os.path.join(_PROJECT_ROOT, "models", model_name + ".pt")
    if getattr(args, "model_state_file", None):
        model_state_file = os.path.abspath(os.path.expanduser(args.model_state_file))
        os.makedirs(os.path.dirname(model_state_file), exist_ok=True)
    print("Sanity Check: stat name : {}".format(model_state_file))
    print("Sanity Check: Is cuda available ? {}".format(torch.cuda.is_available()))

    use_cuda = args.gpu >= 0 and torch.cuda.is_available()

    if args.add_static_graph:
        static_triples = np.array(
            _read_triplets_as_list(
                os.path.join(args.data_root, args.dataset, "e-w-graph.txt"), {}, {}, load_time=False
            )
        )
        num_static_rels = len(np.unique(static_triples[:, 1]))
        num_words = len(np.unique(static_triples[:, 2]))
        static_triples[:, 2] = static_triples[:, 2] + num_nodes 
        static_node_id = torch.from_numpy(np.arange(num_words + data.num_nodes)).view(-1, 1).long().cuda(args.gpu) \
            if use_cuda else torch.from_numpy(np.arange(num_words + data.num_nodes)).view(-1, 1).long()
    else:
        num_static_rels, num_words, static_triples, static_graph = 0, 0, [], None


    # create stat
    model = RecurrentRGCN(args.decoder,
                          args.encoder,
                        num_nodes,
                        num_rels,
                        num_static_rels,
                        num_words,
                        args.n_hidden,
                        args.opn,
                        sequence_len=args.train_history_len,
                        num_bases=args.n_bases,
                        num_basis=args.n_basis,
                        num_hidden_layers=args.n_layers,
                        dropout=args.dropout,
                        self_loop=args.self_loop,
                        skip_connect=args.skip_connect,
                        layer_norm=args.layer_norm,
                        input_dropout=args.input_dropout,
                        hidden_dropout=args.hidden_dropout,
                        feat_dropout=args.feat_dropout,
                        aggregation=args.aggregation,
                        weight=args.weight,
                        pre_weight = args.pre_weight,
                        discount=args.discount,
                        angle=args.angle,
                        use_static=args.add_static_graph,
                        pre_type = args.pre_type,
                        use_cl = args.use_cl,
                        temperature = args.temperature,
                        entity_prediction=args.entity_prediction,
                        relation_prediction=args.relation_prediction,
                        use_cuda=use_cuda,
                        gpu = args.gpu,
                        analysis=args.run_analysis)

    if use_cuda:
        torch.cuda.set_device(args.gpu)
        model.cuda()

    init_ckpt = getattr(args, "init_checkpoint", None)
    if init_ckpt:
        init_ckpt = os.path.abspath(os.path.expanduser(init_ckpt))
        if not os.path.isfile(init_ckpt):
            raise FileNotFoundError("init_checkpoint not found: {}".format(init_ckpt))
        map_loc = torch.device(args.gpu) if use_cuda else torch.device("cpu")
        ckpt = torch.load(init_ckpt, map_location=map_loc)
        state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print("WARN: init_checkpoint missing keys:", len(missing))
        if unexpected:
            print("WARN: init_checkpoint unexpected keys:", len(unexpected))
        print("Loaded init checkpoint:", init_ckpt)

    if args.add_static_graph:
        static_graph = build_sub_graph(len(static_node_id), num_static_rels, static_triples, use_cuda, args.gpu)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    if args.test and os.path.exists(model_state_file):
        mrr_raw, mrr_filter= test(model,
                                train_list+valid_list, 
                                test_list, 
                                num_rels, 
                                num_nodes, 
                                use_cuda, 
                                all_ans_list_test, 
                                all_ans_list_r_test, 
                                model_state_file, 
                                static_graph, 
                                "test")
    elif args.test and not os.path.exists(model_state_file):
        print("--------------{} not exist, Change mode to train and generate stat for testing----------------\n".format(model_state_file))
    else:
        print("----------------------------------------start training----------------------------------------\n")
        profile_memory = bool(getattr(args, "profile_memory", False))
        profile_rows: list[dict] = []
        profile_tag = getattr(args, "profile_memory_tag", None) or (
            "tracer_stage2" if getattr(args, "use_cf", False) else "factual_stage1"
        )
        profile_warmup = int(getattr(args, "profile_memory_warmup", 3))
        profile_out = getattr(args, "profile_memory_out", None)
        gpu_name = _gpu_device_name(args.gpu) if use_cuda else "cpu"
        if profile_memory:
            print(
                "PROFILE_MEMORY enabled | tag={} warmup={} out={} gpu={}".format(
                    profile_tag, profile_warmup, profile_out, gpu_name
                )
            )
        best_mrr = 0
        his_best = 0
        early_stop_patience = int(getattr(args, "early_stop_patience", 5))
        no_mid_valid = bool(getattr(args, "no_mid_valid", False))
        for epoch in range(args.n_epochs):
            model.train()
            losses = []
            losses_e = []
            losses_r = []
            losses_static = []

            idx = [_ for _ in range(len(train_list))]

            for train_sample_num in tqdm(idx):
                if train_sample_num == 0: continue
                output = train_list[train_sample_num:train_sample_num+1]
                if train_sample_num - args.train_history_len<0:
                    input_list = train_list[0: train_sample_num]
                else:
                    input_list = train_list[train_sample_num - args.train_history_len:
                                        train_sample_num]

                subgraph_arr = np.load(
                    os.path.join(args.data_root, args.dataset, "his_graph_for", "train_s_r_{}.npy".format(train_sample_num))
                )
                subgraph_arr_inv = np.load(
                    os.path.join(args.data_root, args.dataset, "his_graph_inv", "train_o_r_{}.npy".format(train_sample_num))
                )
                subg_snap = build_graph(num_nodes, num_rels, subgraph_arr, use_cuda, args.gpu)   # Extract sampled subgraph
                subg_snap_inv = build_graph(num_nodes, num_rels, subgraph_arr_inv, use_cuda, args.gpu)

                inverse_triples = output[0][:, [2, 1, 0]]
                inverse_triples[:, 1] = inverse_triples[:, 1] + num_rels
                que_pair =  e2r(output[0], num_rels)
                que_pair_inv =  e2r(inverse_triples, num_rels)
                # generate history graph
                history_glist = [build_sub_graph(num_nodes, num_rels, snap, use_cuda, args.gpu) for snap in input_list]
                history_glist_cf = None
                cf_sample_mask = None
                cf_sample_mask_inv = None
                cf_entity_mask = None
                if getattr(args, "use_cf", False) and cf_train_list is not None and len(cf_train_list) >= train_sample_num:
                    if train_sample_num - args.train_history_len < 0:
                        input_list_cf = cf_train_list[0:train_sample_num]
                    else:
                        input_list_cf = cf_train_list[
                            train_sample_num - args.train_history_len : train_sample_num
                        ]
                    if len(input_list_cf) == len(input_list):
                        j_need = getattr(args, "cf_min_jaccard", None) is not None or getattr(
                            args, "cf_max_jaccard", None
                        ) is not None
                        skip_cf_graphs = False
                        if j_need:
                            j_val = _cf_window_triple_jaccard(input_list, input_list_cf)
                            lo = args.cf_min_jaccard if args.cf_min_jaccard is not None else 0.0
                            hi = args.cf_max_jaccard if args.cf_max_jaccard is not None else 1.0
                            if not (lo <= j_val <= hi):
                                skip_cf_graphs = True
                        if not skip_cf_graphs:
                            history_glist_cf = [
                                build_sub_graph(num_nodes, num_rels, snap, use_cuda, args.gpu)
                                for snap in input_list_cf
                            ]
                            intervention_entities = _cf_window_intervention_entities(input_list, input_list_cf)
                            cf_entity_mask = _build_cf_entity_mask(num_nodes, intervention_entities, use_cuda, args.gpu)
                            cf_sample_mask = _build_cf_sample_mask(output[0], intervention_entities, use_cuda, args.gpu)
                            cf_sample_mask_inv = _build_cf_sample_mask(inverse_triples, intervention_entities, use_cuda, args.gpu)
                triples = torch.from_numpy(output[0]).long().cuda()
                inverse_triples = torch.from_numpy(inverse_triples).long().cuda()
                cf_branch_active = bool(getattr(args, "use_cf", False) and history_glist_cf is not None)
                if profile_memory and use_cuda:
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                cf_kw = dict(
                    glist_cf=history_glist_cf,
                    use_cf=getattr(args, "use_cf", False) and history_glist_cf is not None,
                    cf_omega=getattr(args, "cf_omega", 1.0),
                    cf_loss_weight=getattr(args, "cf_loss_weight", 1.0),
                    cf_aux_weight=getattr(args, "cf_aux_weight", 0.0),
                    cf_fuse_alpha=getattr(args, "cf_fuse_alpha", 0.0),
                    cf_rank_weight=getattr(args, "cf_rank_weight", 0.0),
                    cf_rank_margin=getattr(args, "cf_rank_margin", 0.05),
                    cf_consistency_weight=getattr(args, "cf_consistency_weight", 0.0),
                    cf_use_dynamic_gate=getattr(args, "cf_use_dynamic_gate", False),
                    cf_entity_mask=cf_entity_mask,
                    cf_residual_weight=getattr(args, "cf_residual_weight", 0.0),
                    cf_quality_tau=getattr(args, "cf_quality_tau", 0.03),
                    cf_quality_beta=getattr(args, "cf_quality_beta", 20.0),
                    cf_no_quality_gate=getattr(args, "cf_no_quality_gate", False),
                )
                for id in range(2):
                    if id % 2 == 0:
                        loss_e, loss_r, loss_static, loss_cl = model.get_loss(
                            que_pair,
                            subg_snap,
                            train_sample_num,
                            history_glist,
                            triples,
                            static_graph,
                            use_cuda,
                            cf_sample_mask=cf_sample_mask,
                            **cf_kw,
                        )
                    else:
                        loss_e, loss_r, loss_static, loss_cl = model.get_loss(
                            que_pair_inv,
                            subg_snap_inv,
                            train_sample_num,
                            history_glist,
                            inverse_triples,
                            static_graph,
                            use_cuda,
                            cf_sample_mask=cf_sample_mask_inv,
                            **cf_kw,
                        )

                    loss = loss_e+ loss_static +loss_cl
                
                    losses.append(loss.item())
                    losses_e.append(loss_e.item())
                    losses_r.append(loss_r.item())
                    losses_static.append(loss_static.item())
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)  # clip gradients
                    optimizer.step()
                    optimizer.zero_grad()
                if profile_memory and use_cuda and train_sample_num > profile_warmup:
                    torch.cuda.synchronize()
                    profile_rows.append(
                        {
                            "setting": profile_tag,
                            "epoch": epoch,
                            "train_sample_num": train_sample_num,
                            "cf_branch_active": int(cf_branch_active),
                            "peak_allocated_gb": round(_bytes_to_gb(torch.cuda.max_memory_allocated()), 4),
                            "peak_reserved_gb": round(_bytes_to_gb(torch.cuda.max_memory_reserved()), 4),
                        }
                    )
                # break
            print("Epoch {:04d} | Ave Loss: {:.4f} | entity-relation-static:{:.4f}-{:.4f}-{:.4f} Best MRR {:.4f} | Model {} "
                  .format(epoch, np.mean(losses), np.mean(losses_e), np.mean(losses_r), np.mean(losses_static), best_mrr, model_name))

            if profile_memory:
                break

            # validation
            if (not no_mid_valid) and epoch and epoch % args.evaluate_every == 0:
                mrr_raw, mrr_filter = test(model, 
                                    train_list, 
                                    valid_list, 
                                    num_rels, 
                                    num_nodes, 
                                    use_cuda, 
                                    all_ans_list_valid, 
                                    all_ans_list_r_valid, 
                                    model_state_file, 
                                    static_graph, 
                                    mode="train")
                
                if not args.relation_evaluation:  # entity prediction evalution
                    if mrr_filter < best_mrr:
                        his_best += 1
                        if epoch >= args.n_epochs:
                            break
                        if early_stop_patience > 0 and his_best >= early_stop_patience:
                            break
                    else:
                        his_best=0
                        best_mrr = mrr_filter
                        torch.save({'state_dict': model.state_dict(), 'epoch': epoch}, model_state_file)
            torch.cuda.empty_cache()
        if profile_memory and profile_out:
            _write_profile_summary(profile_rows, profile_out, profile_tag, gpu_name)
            return None, None
        if not os.path.exists(model_state_file):
            torch.save({'state_dict': model.state_dict(), 'epoch': args.n_epochs - 1}, model_state_file)
        mrr_raw, mrr_filter = test(model,
                            train_list+valid_list,
                            test_list, 
                            num_rels, 
                            num_nodes, 
                            use_cuda, 
                            all_ans_list_test, 
                            all_ans_list_r_test, 
                            model_state_file, 
                            static_graph, 
                            mode="test")
    return mrr_raw, mrr_filter


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TRACER training')

    parser.add_argument("--gpu", type=int, default=0,
                        help="gpu")
    parser.add_argument("--batch-size", type=int, default=1024,
                        help="batch-size")
    parser.add_argument("-d", "--dataset", type=str, default="GDELT",
                        help="dataset to use")
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Directory containing dataset folders (default: <repo>/data)",
    )
    parser.add_argument("--use-cf", action="store_true", default=False,
                        help="Enable counterfactual contrastive branch (requires --cf-train-path)")
    parser.add_argument("--cf-train-path", type=str, default=None,
                        help="Path to cf_train.txt (tab-separated h r o t)")
    parser.add_argument("--cf-omega", type=float, default=1.0,
                        help="Weight omega on exp(sim(z_t,z_cf)) in CF-aware InfoNCE denominator")
    parser.add_argument("--cf-loss-weight", type=float, default=1.0,
                        help="Multiplier for CF contrastive term added on top of original loss_cl")
    parser.add_argument("--cf-aux-weight", type=float, default=0.0,
                        help="Auxiliary NLL weight on CF branch")
    parser.add_argument("--cf-fuse-alpha", type=float, default=0.0,
                        help="Fixed fusion alpha between factual and CF embeddings")
    parser.add_argument("--cf-rank-weight", type=float, default=0.0,
                        help="Intervention-aware margin ranking loss weight")
    parser.add_argument("--cf-rank-margin", type=float, default=0.05,
                        help="Margin for factual-vs-CF ranking loss")
    parser.add_argument("--cf-consistency-weight", type=float, default=0.0,
                        help="KL consistency weight on unaffected samples")
    parser.add_argument("--cf-use-dynamic-gate", action="store_true", default=False,
                        help="Use dynamic gate fusion instead of fixed alpha")
    parser.add_argument("--cf-residual-weight", type=float, default=0.0,
                        help="Residual injection weight for CF delta on entities")
    parser.add_argument("--cf-quality-tau", type=float, default=0.03,
                        help="Quality gate threshold on CF delta magnitude")
    parser.add_argument("--cf-quality-beta", type=float, default=20.0,
                        help="Quality gate sharpness")
    parser.add_argument("--cf-no-quality-gate", action="store_true", default=False,
                        help="Disable per-sample CF quality gate")
    parser.add_argument("--init-checkpoint", type=str, default=None,
                        help="Optional checkpoint to initialize model weights")
    parser.add_argument("--model-state-file", type=str, default=None,
                        help="Override checkpoint save/load path")
    parser.add_argument("--result-file", type=str, default=None,
                        help="Override result CSV path (ablation runs)")
    parser.add_argument("--ablation-tag", type=str, default=None,
                        help="Tag written to ablation result CSV")
    parser.add_argument("--early-stop-patience", type=int, default=5,
                        help="Early stop if valid MRR does not improve for N epochs (0=off)")
    parser.add_argument("--no-mid-valid", action="store_true", default=False,
                        help="Skip mid-training validation; save once before test")
    parser.add_argument(
        "--cf-min-jaccard",
        type=float,
        default=None,
        help="With --use-cf: skip CF for this batch if pooled (h,r,t) Jaccard over the window is below this (omit = no lower bound unless --cf-max-jaccard set)",
    )
    parser.add_argument(
        "--cf-max-jaccard",
        type=float,
        default=None,
        help="With --use-cf: skip CF if pooled Jaccard is above this (omit = no upper bound unless --cf-min-jaccard set)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (torch/numpy/random/dgl) for reproducible runs; omit for non-deterministic",
    )
    parser.add_argument(
        "--profile-memory",
        action="store_true",
        default=False,
        help="Record per-step CUDA peak memory during training; skip validation/final test",
    )
    parser.add_argument(
        "--profile-memory-out",
        type=str,
        default=None,
        help="CSV path for per-step GPU memory records (required with --profile-memory)",
    )
    parser.add_argument(
        "--profile-memory-tag",
        type=str,
        default=None,
        help="Label for profile rows, e.g. factual_stage1 or tracer_stage2",
    )
    parser.add_argument(
        "--profile-memory-warmup",
        type=int,
        default=3,
        help="Skip the first N train steps when aggregating profile stats",
    )
    parser.add_argument("--test", action='store_true', default=False,
                        help="load stat from dir and directly test")
    parser.add_argument("--run-analysis", action='store_true', default=False,
                        help="print log info")
    parser.add_argument("--run-statistic", action='store_true', default=False,
                        help="statistic the result")
    parser.add_argument("--multi-step", action='store_true', default=False,
                        help="do multi-steps inference without ground truth")
    parser.add_argument("--topk", type=int, default=10,
                        help="choose top k entities as results when do multi-steps without ground truth")
    parser.add_argument("--add-static-graph",  action='store_true', default=False,
                        help="use the info of static graph")
    parser.add_argument("--add-rel-word", action='store_true', default=False,
                        help="use words in relaitons")
    parser.add_argument("--relation-evaluation", action='store_true', default=False,
                        help="save model accordding to the relation evalution")
    parser.add_argument("--pre-type",  type=str, default="short",
                        help=["long","short", "all"])
    parser.add_argument("--use-cl",  action='store_true', default=False,
                        help="use the info of  contrastive learning")
    parser.add_argument("--temperature", type=float, default=0.07,
                        help="the temperature of cl")
    # configuration for encoder RGCN stat
    parser.add_argument("--weight", type=float, default=1,
                        help="weight of static constraint")
    parser.add_argument("--pre-weight", type=float, default=0.7,
                        help="weight of entity prediction task")
    parser.add_argument("--discount", type=float, default=1,
                        help="discount of weight of static constraint")
    parser.add_argument("--angle", type=int, default=10,
                        help="evolution speed")
    parser.add_argument("--encoder", type=str, default="uvrgcn", # {uvrgcn,kbat,compgcn}
                        help="method of encoder")
    parser.add_argument("--opn", type=str, default="sub",
                        help="opn of compgcn")
    parser.add_argument("--aggregation", type=str, default="none",
                        help="method of aggregation")
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="dropout probability")
    parser.add_argument("--skip-connect", action='store_true', default=False,
                        help="whether to use skip connect in a RGCN Unit")
    parser.add_argument("--n-hidden", type=int, default=200,
                        help="number of hidden units")
    

    parser.add_argument("--n-bases", type=int, default=100,
                        help="number of weight blocks for each relation")
    parser.add_argument("--n-basis", type=int, default=100,
                        help="number of basis vector for compgcn")
    parser.add_argument("--n-layers", type=int, default=2,
                        help="number of propagation rounds")
    parser.add_argument("--self-loop", action='store_true', default=True,
                        help="perform layer normalization in every layer of gcn ")
    parser.add_argument("--layer-norm", action='store_true', default=False,
                        help="perform layer normalization in every layer of gcn ")
    parser.add_argument("--relation-prediction", action='store_true', default=False,
                        help="add relation prediction loss")
    parser.add_argument("--entity-prediction", action='store_true', default=True,
                        help="add entity prediction loss")
    parser.add_argument("--split_by_relation", action='store_true', default=False,
                        help="do relation prediction")

    # configuration for stat training
    parser.add_argument("--n-epochs", type=int, default=500,
                        help="number of minimum training epochs on each time step")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate")
    parser.add_argument("--grad-norm", type=float, default=1.0,
                        help="norm to clip gradient to")

    # configuration for evaluating
    parser.add_argument("--evaluate-every", type=int, default=1,
                        help="perform evaluation every n epochs")

    # configuration for decoder
    parser.add_argument("--decoder", type=str, default="convtranse",
                        help="method of decoder")
    parser.add_argument("--input-dropout", type=float, default=0.2,
                        help="input dropout for decoder ")
    parser.add_argument("--hidden-dropout", type=float, default=0.2,
                        help="hidden dropout for decoder")
    parser.add_argument("--feat-dropout", type=float, default=0.2,
                        help="feat dropout for decoder")

    # configuration for sequences stat
    parser.add_argument("--train-history-len", type=int, default=10,
                        help="history length")
    parser.add_argument("--test-history-len", type=int, default=20,
                        help="history length for test")
    parser.add_argument("--dilate-len", type=int, default=1,
                        help="dilate history graph")


    args = parser.parse_args()
    print(args)
    args.__dict__["test_history_len"] = args.__dict__["train_history_len"]

    run_experiment(args)



