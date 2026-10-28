"""Threshold tuning and confidence-gated temporal reassignment for Subtask 1."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

import subtask1_complete as original

PERIODS = original.PERIODS


def linkage_for(x: np.ndarray):
    """Compute one average-linkage tree; reuse it for all threshold cuts."""
    if len(x) < 2:
        return None
    distances = pdist(x, metric="cosine")
    if not np.isfinite(distances).all():
        raise ValueError("non-finite distances in embedding matrix")
    distances = np.maximum(distances, 0.0)
    return linkage(distances, method="average")


def cut_tree(tree, n: int, threshold: float) -> np.ndarray:
    if n < 2:
        return np.zeros(n, dtype=np.int32)
    return fcluster(tree, t=float(threshold), criterion="distance").astype(np.int32)


class TemporalContext:
    """One-clustering temporal priors, shared by all temporal candidate settings."""
    def __init__(self, x: np.ndarray, initial: np.ndarray, years: np.ndarray, window: float = 0.5):
        self.initial = np.asarray(initial, dtype=np.int32)
        self.x = x
        self.n = len(initial)
        self.unique, self.codes = np.unique(self.initial, return_inverse=True)
        self.k = len(self.unique)
        if self.k <= 1:
            return
        centroids = np.empty((self.k, x.shape[1]), dtype=np.float32)
        for i in range(self.k):
            centroids[i] = x[self.codes == i].mean(axis=0)
        centroids /= np.maximum(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-12)
        self.sim = x @ centroids.T
        self.own = np.arange(self.n)
        self.sim_own = self.sim[self.own, self.codes]
        without_own = self.sim.copy()
        without_own[self.own, self.codes] = -np.inf
        best_other = without_own.max(axis=1)
        self.ambiguity_gap = np.abs(self.sim_own - best_other)

        years_idx = np.searchsorted(PERIODS, years)
        if np.any(years_idx >= len(PERIODS)) or np.any(PERIODS[years_idx] != years):
            raise ValueError("Unexpected year in temporal context")
        counts = np.zeros((len(PERIODS), self.k), dtype=np.float64)
        np.add.at(counts, (years_idx, self.codes), 1)
        kernel = np.eye(len(PERIODS), dtype=np.float64)
        for t in range(len(PERIODS) - 1):
            kernel[t, t + 1] = window
            kernel[t + 1, t] = window
        aggregate = kernel @ counts
        global_share = counts.sum(axis=0) / self.n
        # Exclude the current occurrence from its estimated local prior.
        local_counts = aggregate[years_idx].copy()
        local_counts[self.own, self.codes] -= 1
        local_total = aggregate.sum(axis=1)[years_idx] - 1
        pseudo = 2.0
        prior = (local_counts + pseudo * global_share[None, :]) / (local_total[:, None] + pseudo)
        self.log_ratio = np.log(np.maximum(prior, 1e-12) / np.maximum(global_share[None, :], 1e-12))

    def predict(self, strength: float, gate: float | None, margin: float = 0.06) -> np.ndarray:
        if float(strength) == 0 or self.k <= 1:
            return self.initial.copy()
        adjusted = self.sim + float(strength) * self.log_ratio
        candidate = adjusted.argmax(axis=1)
        accept = ((candidate != self.codes) &
                  (self.sim[self.own, candidate] >= self.sim_own - margin) &
                  (adjusted[self.own, candidate] > adjusted[self.own, self.codes]))
        if gate is not None:
            accept &= self.ambiguity_gap <= float(gate)
        predicted = self.initial.copy()
        predicted[accept] = self.unique[candidate[accept]]
        return predicted


def candidate_specs(strengths, gates):
    specs = [("static", 0.0, None)]
    for strength in strengths:
        if strength <= 0:
            continue
        specs.append(("temporal_original", float(strength), None))
        for gate in gates:
            specs.append(("temporal_gated", float(strength), float(gate)))
    return specs


def score_case(word, x, gold, years, initial, pred, threshold, method, strength, gate, gold_change):
    p, r, f = original.bcubed(gold, pred)
    return {
        "word": word, "threshold": threshold, "method": method,
        "strength": strength, "gate": "none" if gate is None else str(gate),
        "bcubed_f1": f, "bcubed_precision": p, "bcubed_recall": r,
        "gold_change_observed": gold_change[0], "gold_change_adjacent": gold_change[1],
        "pred_change_observed": original.jsd_change(pred, years, "observed"),
        "pred_change_adjacent": original.jsd_change(pred, years, "adjacent"),
        "n_clusters": int(len(np.unique(pred))),
        "n_changed": int(np.count_nonzero(pred != initial)),
        "n_sentences": int(len(years)),
        "n_missing_periods": int(len(PERIODS) - len(np.unique(years))),
    }


def evaluate_many(df, mmap, positions, words, thresholds, specs, audit_equivalence=0):
    rows = []
    for j, word in enumerate(words, 1):
        loc = positions[word]
        group = df.iloc[loc]
        x = original.normalized(mmap[loc])
        gold = group["sense_id"].to_numpy()
        years = group["year"].to_numpy(dtype=np.int64)
        gold_change = (original.jsd_change(gold, years, "observed"),
                       original.jsd_change(gold, years, "adjacent"))
        tree = linkage_for(x)
        for t in thresholds:
            initial = cut_tree(tree, len(x), t)
            if j <= audit_equivalence:
                from sklearn.metrics import adjusted_rand_score
                old_clusters = original.cluster(x, t)
                if adjusted_rand_score(initial, old_clusters) < 1 - 1e-12:
                    raise AssertionError(f"New linkage cuts differ from original sklearn AHC: {word=} {t=}")
            context = None
            for method, strength, gate in specs:
                if method == "static":
                    pred = initial
                else:
                    if context is None:
                        context = TemporalContext(x, initial, years)
                    pred = context.predict(strength, gate)
                rows.append(score_case(word, x, gold, years, initial, pred,
                                       t, method, strength, gate, gold_change))
        if j % 50 == 0 or j == len(words):
            print(f"  processed {j}/{len(words)} words", flush=True)
    return pd.DataFrame(rows)


def summarize(rows):
    summaries = []
    for keys, group in rows.groupby(["threshold", "method", "strength", "gate"], sort=False):
        threshold, method, strength, gate = keys
        summaries.append({
            "threshold": float(threshold), "method": method,
            "strength": float(strength), "gate": gate,
            "words": len(group),
            "bcubed_f1": group.bcubed_f1.mean(),
            "bcubed_precision": group.bcubed_precision.mean(),
            "bcubed_recall": group.bcubed_recall.mean(),
            "rho_observed": original.score_rho(group.pred_change_observed, group.gold_change_observed),
            "rho_adjacent": original.score_rho(group.pred_change_adjacent, group.gold_change_adjacent),
            "words_missing_periods": int((group.n_missing_periods > 0).sum()),
            "changed_sentences": int(group.n_changed.sum()),
            "sentences": int(group.n_sentences.sum()),
        })
    return pd.DataFrame(summaries)


def choose(summary, metric):
    # Break ties in favor of the simpler model.
    z = summary.copy()
    z["complexity"] = z["method"].map({"static":0, "temporal_gated":1, "temporal_original":2})
    z = z.sort_values([metric, "complexity", "strength", "threshold"],
                      ascending=[False, True, True, True])
    return z.iloc[0]


def to_spec(row):
    return (float(row.threshold), str(row.method), float(row.strength),
            None if row.gate == "none" else float(row.gate))


def make_config_set(*items):
    return list(dict.fromkeys(items))


def paired_bootstrap(rows, model_a, model_b, metric, n_boot=1000, seed=42):
    """Per-word paired bootstrap; same sampled words in both methods.

    Delta: model_a - model_b. For rho, resample complete word triplets.
    """
    if metric not in ("bcubed_f1", "rho_observed"):
        raise ValueError(metric)
    a = rows[rows.config == model_a].set_index("word").sort_index()
    b = rows[rows.config == model_b].set_index("word").sort_index()
    if len(a) == 0 or len(b) == 0 or not a.index.equals(b.index):
        raise ValueError("Models must be paired on identical words")
    rng = np.random.default_rng(seed)
    n = len(a)
    if metric == "bcubed_f1":
        u = a.bcubed_f1.to_numpy() - b.bcubed_f1.to_numpy()
        point = float(np.mean(u))
        sims = np.mean(u[rng.integers(0, n, size=(n_boot, n))], axis=1)
    else:
        gold = a.gold_change_observed.to_numpy()
        if not np.allclose(gold, b.gold_change_observed):
            raise ValueError("Gold change scores differ between methods")
        pa = a.pred_change_observed.to_numpy()
        pb = b.pred_change_observed.to_numpy()
        point = original.score_rho(pa, gold) - original.score_rho(pb, gold)
        sims = np.empty(n_boot, dtype=np.float64)
        for z in range(n_boot):
            idx = rng.integers(0, n, size=n)
            sims[z] = original.score_rho(pa[idx], gold[idx]) - original.score_rho(pb[idx], gold[idx])
    lo, hi = np.quantile(sims, [0.025, 0.975])
    return {"model":model_a, "control":model_b, "metric":metric, "delta":float(point),
            "ci_95_low":float(lo),"ci_95_high":float(hi),"bootstrap_resamples":n_boot}


def evaluate_selected(df, mmap, positions, words, configs, audit_equivalence=0):
    specs_by_t = {}
    for t, method, strength, gate in configs:
        specs_by_t.setdefault(float(t), []).append((method, strength, gate))
    all_rows = []
    for j, word in enumerate(words,1):
        loc = positions[word]
        group = df.iloc[loc]
        x = original.normalized(mmap[loc])
        gold = group.sense_id.to_numpy()
        years = group.year.to_numpy(dtype=np.int64)
        gold_change = (original.jsd_change(gold, years, "observed"),
                       original.jsd_change(gold, years, "adjacent"))
        tree = linkage_for(x)
        for t, specs in specs_by_t.items():
            ini = cut_tree(tree,len(x),t)
            if j <= audit_equivalence:
                from sklearn.metrics import adjusted_rand_score
                assert adjusted_rand_score(ini,original.cluster(x,t)) > 1-1e-12
            ctx = None
            for method, strength, gate in specs:
                if method == "static":
                    pred = ini
                else:
                    if ctx is None:
                        ctx = TemporalContext(x, ini, years)
                    pred = ctx.predict(strength, gate)
                item = score_case(word,x,gold,years,ini,pred,t,method,strength,gate,gold_change)
                item["config"] = key((t, method, strength, gate))
                all_rows.append(item)
        if j % 50 == 0 or j == len(words):
            print(f"  evaluated {j}/{len(words)} words",flush=True)
    return pd.DataFrame(all_rows)


def key(cfg):
    t, m, s, g = cfg
    return f"thr={t:g}|{m}|str={s:g}|gate={'none' if g is None else f'{g:g}'}"


def write_json(path,obj):
    path.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding="utf-8")


def charts(static_sum, temporal_sum, comparison, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = static_sum.sort_values("threshold")
    for metric in ("bcubed_f1","rho_observed"):
        fig,ax=plt.subplots(figsize=(6,4))
        ax.plot(s.threshold,s[metric],"o-")
        ax.set(xlabel="AHC cosine threshold",ylabel=metric,
               title=f"Training sweep: {metric}")
        fig.tight_layout()
        fig.savefig(output/f"train_threshold_{metric}.png",dpi=170)
        plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4.5))
    z=temporal_sum.copy()
    for method,part in z.groupby("method"):
        if method=="static": continue
        if method=="temporal_gated":
            for gate,gg in part.groupby("gate"):
                ax.plot(gg.strength,gg.bcubed_f1,"o-",label=f"gated {gate}")
        else:
            ax.plot(part.strength,part.bcubed_f1,"o-",label="original")
    ax.set(xlabel="temporal strength",ylabel="training macro BCubed F1",
           title="Temporal strength and uncertainty gating")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output/"train_temporal_gating.png",dpi=170)
    plt.close(fig)


def run(args):
    started=time.monotonic()
    root=Path(args.repo_root).expanduser().resolve()
    datadir=Path(args.data_dir).expanduser().resolve()
    out=Path(args.output_dir).expanduser().resolve()
    out.mkdir(parents=True,exist_ok=True)
    emb=datadir/"embeddings.npy"
    parq=datadir/"embeddings_sentences.parquet"
    if not emb.exists() or not parq.exists():
        raise FileNotFoundError("Expected embeddings.npy and embeddings_sentences.parquet at data-dir")
    df=pd.read_parquet(parq)
    split=pd.read_csv(root/"splits/word_split.csv")
    effective, split_record=original.reconcile_split(df,split,out)
    original.verify_data(df,emb,effective)
    if args.train_words <= 0: raise ValueError("--train-words must be positive")
    train=effective.loc[effective.split=="train","lemma"].astype(str).to_numpy()
    dev=effective.loc[effective.split=="dev","lemma"].astype(str).tolist()
    test=effective.loc[effective.split=="test","lemma"].astype(str).tolist()
    chosen=np.random.default_rng(args.seed).choice(train,size=min(args.train_words,len(train)),replace=False).tolist()
    mmap=np.load(emb,mmap_mode="r")
    positions=df.groupby("lemma",sort=False).indices

    print("\nSTAGE 1: wider threshold sweep on TRAIN only",flush=True)
    static_train=evaluate_many(df,mmap,positions,chosen,args.thresholds,[("static",0.0,None)],audit_equivalence=args.audit_equivalence)
    static_train.to_csv(out/"train_threshold_per_word.csv",index=False)
    static_sum=summarize(static_train)
    static_sum.to_csv(out/"train_threshold_sweep.csv",index=False)
    static_for_f1=choose(static_sum,"bcubed_f1")
    static_for_rho=choose(static_sum,"rho_observed")
    print(static_sum.to_string(index=False,float_format=lambda x:f"{x:.4f}"))
    print("Static selected thresholds (F1, rho):",static_for_f1.threshold,static_for_rho.threshold,flush=True)

    print("\nSTAGE 2: temporal methods on TRAIN only",flush=True)
    t_values=sorted(set([float(static_for_f1.threshold),float(static_for_rho.threshold)]))
    temporal_train=evaluate_many(df,mmap,positions,chosen,t_values,candidate_specs(args.strengths,args.gates))
    temporal_train.to_csv(out/"train_temporal_per_word.csv",index=False)
    temporal_sum=summarize(temporal_train)
    temporal_sum.to_csv(out/"train_temporal_sweep.csv",index=False)
    # Select each task configuration at its training-selected threshold.
    select_f1=choose(temporal_sum[temporal_sum.threshold==float(static_for_f1.threshold)],"bcubed_f1")
    select_rho=choose(temporal_sum[temporal_sum.threshold==float(static_for_rho.threshold)],"rho_observed")
    print("Selected for 1a:",to_spec(select_f1),"selected for 1b:",to_spec(select_rho),flush=True)

    cfg_base=(0.35,"static",0.0,None)
    cfg_static_f1=to_spec(static_for_f1)
    cfg_static_rho=to_spec(static_for_rho)
    cfg_f1=to_spec(select_f1)
    cfg_rho=to_spec(select_rho)
    cfgs=make_config_set(cfg_base,cfg_static_f1,cfg_static_rho,cfg_f1,cfg_rho)
    # Optionally retain the reference temporal model for comparison.
    if args.include_reference_temporal:
        cfgs=make_config_set(*cfgs,(float(static_for_f1.threshold),"temporal_original",0.2,None))
    print("\nSTAGE 3: frozen configurations on DEV",flush=True)
    dev_rows=evaluate_selected(df,mmap,positions,dev,cfgs,audit_equivalence=min(5,args.audit_equivalence))
    dev_rows.to_csv(out/"dev_per_word.csv",index=False)
    dev_summary=summarize(dev_rows)
    dev_summary["config"]=[key((r.threshold,r.method,r.strength,None if r.gate=="none" else float(r.gate))) for r in dev_summary.itertuples()]
    dev_summary.to_csv(out/"dev_comparison.csv",index=False)
    print(dev_summary.to_string(index=False,float_format=lambda x:f"{x:.4f}"))

    deltas=[]
    comparisons=[(cfg_static_f1,cfg_base,"bcubed_f1"),(cfg_f1,cfg_static_f1,"bcubed_f1"),
                 (cfg_static_rho,cfg_base,"rho_observed"),(cfg_rho,cfg_static_rho,"rho_observed")]
    for cfg_a,cfg_b,metric in comparisons:
        if cfg_a==cfg_b: continue
        deltas.append(paired_bootstrap(dev_rows,key(cfg_a),key(cfg_b),metric,n_boot=args.bootstrap,seed=args.seed))
    pd.DataFrame(deltas,columns=["model","control","metric","delta","ci_95_low","ci_95_high","bootstrap_resamples"]).to_csv(out/"dev_bootstrap_deltas.csv",index=False)
    # Compare word-level F1 against the tuned static model.
    paired=dev_rows[dev_rows.config.isin([key(cfg_static_f1),key(cfg_f1)])].pivot_table(index="word",columns="config",values="bcubed_f1")
    if key(cfg_f1) != key(cfg_static_f1):
        analysis=pd.DataFrame({"word":paired.index,"delta_f1":(paired[key(cfg_f1)]-paired[key(cfg_static_f1)]).to_numpy()})
        analysis.sort_values("delta_f1").to_csv(out/"dev_word_changes_1a.csv",index=False)
    charts(static_sum,temporal_sum,dev_summary,out)

    settings={"static_threshold_1a":float(static_for_f1.threshold),
              "static_threshold_1b":float(static_for_rho.threshold),
              "selected_1a":dict(threshold=cfg_f1[0],method=cfg_f1[1],strength=cfg_f1[2],gate=cfg_f1[3]),
              "selected_1b":dict(threshold=cfg_rho[0],method=cfg_rho[1],strength=cfg_rho[2],gate=cfg_rho[3]),
              "evaluated_configurations":[key(c) for c in cfgs]}
    write_json(out/"selected_train_parameters.json",settings)
    if args.final_test:
        print("\nSTAGE 4: explicitly requested TEST evaluation (do not tune after this)",flush=True)
        test_rows=evaluate_selected(df,mmap,positions,test,cfgs)
        test_rows.to_csv(out/"test_per_word.csv",index=False)
        test_sum=summarize(test_rows)
        test_sum.to_csv(out/"test_comparison.csv",index=False)
        print(test_sum.to_string(index=False,float_format=lambda x:f"{x:.4f}"))
    else:
        print("TEST was NOT evaluated. (Use --final-test only once methods are frozen.)",flush=True)
    manifest={"timestamp_utc":datetime.now(timezone.utc).isoformat(),
              "source":"synthetic single-label generated sense annotations; NOT official scorer",
              "train_sample_words":len(chosen),"seed":args.seed,"train_words":chosen,
              "thresholds":list(map(float,args.thresholds)),"strengths":list(map(float,args.strengths)),
              "gates":list(map(float,args.gates)),"split_reconciliation":split_record,
              "selected":settings,"dev_words":len(dev),"test_evaluated":bool(args.final_test),
              "JSD":"rho_observed reproduces original skip-missing convention; rho_adjacent is diagnostic only",
              "sampling":"No assertion that generated counts match/mismatch metadata frequencies",
              "seconds_elapsed":round(time.monotonic()-started,2)}
    write_json(out/"run_manifest.json",manifest)
    print("Results written to",out,flush=True)


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root",required=True)
    p.add_argument("--data-dir",required=True)
    p.add_argument("--output-dir",required=True)
    p.add_argument("--train-words",type=int,default=500)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--thresholds",type=float,nargs="+",default=[0.25,0.30,0.35,0.4,0.45,0.5,0.55,0.6,0.65])
    p.add_argument("--strengths",type=float,nargs="+",default=[0.05,0.1,0.2,0.35])
    p.add_argument("--gates",type=float,nargs="+",default=[0.025,0.05,0.10])
    p.add_argument("--bootstrap",type=int,default=1000)
    p.add_argument("--audit-equivalence",type=int,default=10)
    p.add_argument("--include-reference-temporal",action="store_true")
    p.add_argument("--final-test",action="store_true")
    return p.parse_args()

if __name__=="__main__":
    run(parse_args())
