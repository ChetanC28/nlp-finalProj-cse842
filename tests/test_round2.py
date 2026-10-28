"""Unit tests for threshold equivalence and experiment execution."""
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
import subtask1_complete as old
import subtask1_round2 as new


def test_equivalence():
    rng = np.random.default_rng(3)
    for n in (1,2,6,40):
        x=old.normalized(rng.normal(size=(n,24)).astype('float32'))
        tree=new.linkage_for(x)
        for t in (0.25, 0.35, 0.45, 0.65, 1.2):
            a=old.cluster(x,t)
            b=new.cut_tree(tree,n,t)
            assert adjusted_rand_score(a,b)>1-1e-12, (n,t,a,b)
            if n>2:
                years=np.resize(new.PERIODS,n)
                context=new.TemporalContext(x,b,years)
                predicted=context.predict(.2,None)
                reference=old.smooth_reassign(x,b,years,.2)
                assert np.array_equal(predicted,reference),(n,t)
                for gate in (.025,.05,.1):
                    c=context.predict(.2,gate)
                    assert ((c!=b) <= (reference!=b)).all()


def test_end_to_end():
    rng=np.random.default_rng(23)
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp)/'repo';data=Path(tmp)/'data';out=Path(tmp)/'out'
        (root/'splits').mkdir(parents=True);data.mkdir()
        metadata=[];emb=[];splits=[]
        years=[1810,1830,1850,1870,1890]
        for j in range(18):
            lemma=f'word{j:02d}'
            label='train' if j<11 else 'dev' if j<16 else 'test'
            splits.append((lemma,label))
            for i in range(18):
                sense=f'sense{i%3}'
                metadata.append({'lemma':lemma,'sense_id':sense,'year':years[(i+j)%len(years)],
                                 'text':f'{lemma} example {i}','start':0,'end':len(lemma)})
                z=rng.normal(0,.35,16);z[(i%3)*3]+=2.0
                emb.append(z)
        splits.append(('missing','train'))
        pd.DataFrame(splits,columns=['lemma','split']).to_csv(root/'splits/word_split.csv',index=False)
        pd.DataFrame(metadata).to_pickle(data/'embeddings_sentences.parquet')
        # Mock Parquet reads to avoid a pyarrow requirement for test data.
        read_parquet = pd.read_parquet
        pd.read_parquet = pd.read_pickle
        np.save(data/'embeddings.npy',np.array(emb,dtype='float16'))
        class A: pass
        a=A();a.repo_root=str(root);a.data_dir=str(data);a.output_dir=str(out)
        a.train_words=9;a.seed=42;a.thresholds=[.25,.35,.45,.65];a.strengths=[.1,.2];a.gates=[.05,.1]
        a.bootstrap=25;a.audit_equivalence=5;a.include_reference_temporal=True;a.final_test=False
        try:
            new.run(a)
        finally:
            pd.read_parquet = read_parquet
        assert (out/'dev_comparison.csv').exists()
        assert (out/'dev_bootstrap_deltas.csv').exists()
        assert (out/'selected_train_parameters.json').exists()
        assert not (out/'test_comparison.csv').exists()
        assert 'missing' in (out/'split_reconciliation.json').read_text()
        assert len(pd.read_csv(out/'dev_comparison.csv'))>=2
        print('synthetic end-to-end completed')

if __name__=='__main__':
    test_equivalence()
    test_end_to_end()
    print('All Round 2 tests passed')
