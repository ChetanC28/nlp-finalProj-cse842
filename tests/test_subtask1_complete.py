"""Tests for clustering metrics, temporal handling, and split validation."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
import pandas as pd

from subtask1_complete import bcubed, jsd_change, smooth_reassign, evaluate, run, reconcile_split

class TestSubtask1(unittest.TestCase):
    def test_bcubed(self):
        self.assertAlmostEqual(bcubed(['a','a','b','b'], [0,0,1,1])[2], 1.0)
        self.assertAlmostEqual(bcubed(['a','a','b','b'], [0,0,0,0])[2], 2.0/3.0)

    def test_missing_period_modes(self):
        years=np.array([1810,1810,1850,1850])
        labels=np.array([0,0,1,1])
        self.assertAlmostEqual(jsd_change(labels, years, 'observed'), 1.0)
        self.assertAlmostEqual(jsd_change(labels, years, 'adjacent'), 0.0)

    def test_strength_zero(self):
        x=np.eye(4, dtype=np.float32)
        labels=np.array([0,0,1,1])
        years=np.array([1810,1830,1850,1870])
        self.assertTrue(np.array_equal(labels, smooth_reassign(x, labels, years, 0.0)))

    def test_split_reconciliation(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            df = pd.DataFrame({"lemma": ["a", "b", "c"]})
            split = pd.DataFrame({"lemma": ["a", "b", "jock"],
                                  "split": ["train", "dev", "test"]})
            valid, report = reconcile_split(df, split, out)
            self.assertEqual(set(valid.lemma), {"a", "b"})
            self.assertEqual(report["missing_from_embeddings"], [{"lemma": "jock", "split": "test"}])
            self.assertEqual(report["without_split_assignment"], ["c"])
            self.assertTrue((out / "split_reconciliation.json").exists())

    def test_full_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); (root/'splits').mkdir(); (root/'cache').mkdir()
            words=[('u','train'),('v','train'),('w','dev'),('x','dev'),('y','test'),('z','test')]
            pd.DataFrame(words,columns=['lemma','split']).to_csv(root/'splits/word_split.csv',index=False)
            rng=np.random.default_rng(35)
            all_rows=[]; all_vec=[]
            for wi,(word,part) in enumerate(words):
                for t in [1810,1830,1870,1890]:
                    for k in range(2):
                        for n in range(4):
                            text=f'The {word} was here.'
                            all_rows.append(dict(lemma=word,sense_id=f's{k}',year=t,text=text,start=4,end=5))
                            vec=np.zeros(8,dtype=np.float32); vec[k]=2
                            if k == 1 and t == 1830: vec[k] = 0.7
                            all_vec.append(vec+rng.normal(0,0.2,8))
            # Mock the Parquet reader; the production path uses actual Parquet files.
            from unittest.mock import patch
            pd.DataFrame(all_rows).to_pickle(root/'cache/embeddings_sentences.parquet')
            np.save(root/'cache/embeddings.npy',np.asarray(all_vec,np.float16))
            args=SimpleNamespace(data_dir=str(root/'cache'),repo_root=str(root),output_dir=str(root/'results'),train_words=2,seed=42,thresholds=[.25,.35,.45],strengths=[0,.05],final_test=True)
            with patch('subtask1_complete.pd.read_parquet', side_effect=pd.read_pickle):
                run(args)
            files=['baseline_dev_summary.csv','train_threshold_sweep.csv','train_temporal_sweep.csv','dev_comparison.csv','test_comparison.csv','run_manifest.json']
            for filename in files:
                self.assertTrue((root/'results'/filename).exists(),filename)
            self.assertEqual(len(pd.read_csv(root/'results/baseline_dev_per_word.csv')),2)

if __name__ == '__main__':
    unittest.main()
