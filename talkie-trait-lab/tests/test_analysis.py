import json
import pytest
from trait_lab.io import write,sha
from trait_lab.analyze import contrast

def fixture(path,high,interface='bare'):
    path.mkdir()
    rows=[]
    for i,selected in enumerate(high):
        sid=str(i//2)
        rows.append({'example':{'id':str(i),'prompt':f'Q{sid}','choices':['high','low'],
            'answer_index':0,'category':'Openness','polarity':'good','source_scenario_id':sid,
            'group_id':sid},'high_selected':selected,'bits_per_byte':[1.,2.] if selected else [2.,1.]})
    (path/'results.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    write(path/'summary.json',{'interface':interface,'rows':len(rows),'results_sha256':sha(path/'results.jsonl')})

def test_paired_bootstrap_keeps_both_pairs_in_each_scenario(tmp_path):
    a=tmp_path/'a';b=tmp_path/'b';fixture(a,[1,1,0,0]);fixture(b,[0,0,1,1])
    result=contrast(a,b,tmp_path/'out.csv',replicates=1000)[0]
    assert result['rows']==4 and result['delta_pp']==0
    assert result['low_to_high']==2 and result['high_to_low']==2
    assert result['ci_low_pp']==-100 and result['ci_high_pp']==100

def test_analysis_rejects_cross_interface_delta(tmp_path):
    a=tmp_path/'a';b=tmp_path/'b';fixture(a,[1,1],interface='bare');fixture(b,[0,0],interface='chat')
    with pytest.raises(ValueError,match='separately'):contrast(a,b,tmp_path/'out.csv')
