import pytest
from vllm_tuner.data import digest, write_json, prepare_dataset, freeze_protocol, prompt_for

def test_immutable_and_stable_hash(tmp_path):
    p=tmp_path/'x.json'; write_json(p,{'a':1},immutable=True); write_json(p,{'a':1},immutable=True)
    with pytest.raises(FileExistsError): write_json(p,{'a':2},immutable=True)
    assert digest({'a':1,'b':2})==digest({'b':2,'a':1})

def test_test_data_requires_final_freeze(tmp_path):
    with pytest.raises(ValueError): prepare_dataset(tmp_path,'test')
    with pytest.raises(ValueError): freeze_protocol(tmp_path/'freeze.json',{},[{'source':'synthetic','metrics':{'feasible':True}}],{})

def test_prompt_never_receives_gold():
    class Tokenizer:
        def apply_chat_template(self,messages,**kwargs):
            assert kwargs['enable_thinking']; return messages[0]['content']
    p=prompt_for('What is 2+2?',Tokenizer())
    assert 'boxed' in p and 'What is 2+2?' in p


def test_test_download_rejects_dummy_freeze_before_network(tmp_path):
    from vllm_tuner.data import DATA_REVISION
    write_json(tmp_path/'dummy.json',{'data_revision':DATA_REVISION})
    with pytest.raises((ValueError,KeyError)):
        prepare_dataset(tmp_path,'test',freeze_path=tmp_path/'dummy.json')
