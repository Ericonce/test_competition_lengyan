"""External layouts must become standalone class-folder data before training."""
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
import yaml

from pulsecls import PulseCLS
from pulsecls.cfg import REPO_ROOT, get_cfg, load_yaml
from pulsecls.data.dataset import load_sample, load_split_arrays, read_sequence_csv
from pulsecls.data.file_index import load_inventory, record_at
from pulsecls.data.selection import select_records
from pulsecls.data.sources import input_config
from pulsecls.data.conversion import convert_dataset


FORMATS = ("label_file", "filename", "table")
FEATURES = ["CF_MHz", "PW_us", "PRI_us"]


def make_inputs(root, kind):
    root.mkdir(parents=True, exist_ok=True)
    rows, labels, expected = [], [], []
    for label in range(2):
        for sample in range(6):
            length = 20 + sample * 4
            raw = np.column_stack((3000 + 100 * label + sample + np.arange(length),
                                   10 + label + np.arange(length) * .1,
                                   500 + sample + np.arange(length) * .5))
            name = f"class{label}_sample{sample}.csv"
            expected.append((label, raw))
            if kind != "table":
                np.savetxt(root/name, raw, delimiter=",", header=",".join(FEATURES), comments="")
            labels.append((name, label))
            rows.extend((f"c{label}s{sample}", label, pulse, *values) for pulse, values in enumerate(raw))
    options = {"format": kind}
    if kind == "label_file":
        with (root/'labels.csv').open('w', newline='', encoding='utf-8-sig') as stream:
            writer = csv.writer(stream)
            writer.writerow(['filename','label'])
            writer.writerows(reversed(labels))
    if kind == "table":
        # Interleaving and reverse pulse order exercise multi-span grouping and ordering.
        rows.sort(key=lambda row: (-row[2], row[0]))
        with (root/'samples.csv').open('w', newline='', encoding='utf-8-sig') as stream:
            writer = csv.writer(stream)
            writer.writerow(['sample_id','label','pulse_index',*FEATURES])
            writer.writerows(rows)
        options['order_column'] = 'pulse_index'
    cfg = {"root": str(root), "num_classes": 2, "features": FEATURES, "sequence_length": 32,
           "length_policy": {"short": "zero", "long": "head"}, "input": options,
           "selection": {"sample_ratio": 1., "train_ratio": .75, "val_ratio": .25,
                         "seed": 2026, "stratified": True}, "cache": True, "progress": False}
    return cfg, expected


def converted_config(description):
    cfg = load_yaml(description)
    cfg['root'] = str(description.parent)
    return cfg


@pytest.mark.parametrize('kind', FORMATS)
def test_conversion_preserves_samples_and_uses_existing_class_folder_pipeline(tmp_path, kind):
    source = tmp_path/'raw'
    cfg, expected = make_inputs(source, kind)
    originals = {p.name:p.read_bytes() for p in source.glob('*.csv')}
    description = convert_dataset(cfg, tmp_path/'converted', progress=False)
    converted = converted_config(description)
    assert converted['input'] == {'format':'class_folders'}
    assert load_yaml(description)['root'] == '.'
    assert load_yaml(description)['all'] == '.'
    inventory, _, _ = load_inventory(description.parent, converted)
    assert [(g['label'], g['count']) for g in inventory] == [(0,6),(1,6)]
    parts, info = select_records(converted)
    assert (info['total_count'],info['train_count'],info['val_count']) == (12,9,3)
    assert not set(parts['train']) & set(parts['val'])
    assert select_records(converted)[0] == parts
    by_start = {float(raw[0,0]):(label,raw) for label,raw in expected}
    for path, label in parts['all']:
        assert isinstance(path,Path) and path.parent.name == str(label) and path.is_file()
        raw = read_sequence_csv(path)
        expected_label, expected_raw = by_start[float(raw[0,0])]
        assert label == expected_label
        np.testing.assert_array_equal(raw,expected_raw)
        if kind != 'table':
            assert path.read_bytes() == originals[path.name]
        else:
            assert next(csv.reader(path.open(encoding='utf-8'))) == FEATURES
    train,train_y,train_paths = load_split_arrays(converted,'train')
    val,val_y,val_paths = load_split_arrays(converted,'val')
    assert train.shape == (9,32,3) and val.shape == (3,32,3)
    assert set(train_y) == set(val_y) == {0,1}
    assert not set(train_paths) & set(val_paths)
    with patch('pulsecls.data.dataset.load_sample', side_effect=AssertionError('CSV reread despite valid cache')):
        np.testing.assert_array_equal(load_split_arrays(converted,'train')[0],train)
    assert {p.name:p.read_bytes() for p in source.glob('*.csv')} == originals
    manifest = list(csv.DictReader((description.parent/'conversion_manifest.csv').open(encoding='utf-8-sig')))
    assert len(manifest) == 12 and len({row['output_path'] for row in manifest}) == 12
    assert sorted(int(row['num_pulses']) for row in manifest) == sorted(len(raw) for _,raw in expected)
    summary = json.loads((description.parent/'conversion_summary.json').read_text(encoding='utf-8'))
    assert summary['class_counts'] == {'0':6,'1':6} and summary['num_samples'] == 12
    assert summary['input']['format'] == kind


@pytest.mark.parametrize('kind', FORMATS)
def test_raw_layouts_are_rejected_by_training_and_inventory(tmp_path,kind):
    cfg,_ = make_inputs(tmp_path/'raw',kind)
    pipeline = {'task':'classify','output_dir':str(tmp_path/'run'),'data_cfg':cfg,'model_cfg':{'type':'cnn1d'}}
    for operation in (lambda:select_records(cfg), lambda:load_inventory(cfg['root'],cfg),
                      lambda:load_split_arrays(cfg,'train'), lambda:get_cfg(pipeline)):
        with pytest.raises(ValueError,match='convert_dataset.py'):
            operation()


@pytest.mark.parametrize('override', [
    {'data_input':{'format':'filename'}},
    {'data_cfg':{'input':{'format':'label_file'}}},
    {'data_cfg.input.format':'table'},
])
def test_previous_direct_input_overrides_require_conversion(tmp_path,override):
    pipeline={'task':'classify','output_dir':str(tmp_path/'run'),
              'data_cfg':{'root':str(tmp_path),'num_classes':2},'model_cfg':{'type':'cnn1d'}}
    with pytest.raises(ValueError,match='convert_dataset.py'):
        get_cfg(pipeline,override)


@pytest.mark.parametrize('kind',FORMATS)
def test_templates_are_conversion_inputs_not_training_choices(kind):
    template=REPO_ROOT/f'configs/data/{kind}.yaml'
    assert input_config(load_yaml(template)['input'])['format'] == kind
    with pytest.raises(ValueError,match='convert_dataset.py'):
        get_cfg(REPO_ROOT/'configs/pipeline/cnn_softmax_synthetic.yaml',{'data':template})


@pytest.mark.parametrize('delimiter,header', [(',',True),('\t',True),('whitespace',False)])
def test_label_table_csv_tsv_and_headerless_txt(tmp_path,delimiter,header):
    cfg,_ = make_inputs(tmp_path/'raw','label_file')
    rows = ['path,target'] if header else []
    rows += [f'class{label}_sample{i}.csv,{label+1}' for label in range(2) for i in range(6)]
    separator = ' ' if delimiter=='whitespace' else delimiter
    (tmp_path/'raw/labels.txt').write_text('\n'.join(line.replace(',',separator) for line in rows),encoding='utf-8')
    cfg['input'].update(labels_file='labels.txt',pattern='class*.csv',filename_column='path',label_column='target',
                        header=header,delimiter=delimiter,label_offset=-1)
    result=convert_dataset(cfg,tmp_path/'converted',progress=False)
    inventory,_,_=load_inventory(result.parent,converted_config(result))
    assert [(g['label'],g['count']) for g in inventory] == [(0,6),(1,6)]


@pytest.mark.parametrize('kind', FORMATS)
def test_label_mapping_is_saved_and_reverses_target_directories(tmp_path,kind):
    cfg,_=make_inputs(tmp_path/'raw',kind)
    cfg['input'].update(label_map={'0':1,'1':0},label_offset=99)
    result=convert_dataset(cfg,tmp_path/'converted',progress=False)
    inventory,_,_=load_inventory(result.parent,converted_config(result))
    for i in range(12):
        ref,label=record_at(inventory,i)
        assert label == (0 if read_sequence_csv(ref)[0,0]>=3100 else 1)
    assert json.loads((result.parent/'conversion_summary.json').read_text())['input']['label_map'] == {'0':1,'1':0}


@pytest.mark.parametrize('order', [None,'pulse_index'])
def test_table_composite_multiline_ids_aliases_encoding_and_order(tmp_path,order):
    source=tmp_path/'raw'
    source.mkdir()
    path=source/'samples.tsv'
    with path.open('w',newline='',encoding='gb18030') as stream:
        writer=csv.writer(stream,delimiter='\t')
        writer.writerow(['sample_id','label','pulse_index','载频(MHz)','脉宽(us)','PRI'])
        writer.writerows([['shared\nID',0,1,'10.123456789012345',2,100],
                          ['shared\nID',1,0,20,3,200],['shared\nID',0,0,11,4,101]])
    cfg={'root':str(source),'num_classes':2,'input':{'format':'table','table_file':'samples.tsv',
         'sample_id_columns':['label','sample_id'],'delimiter':'\t','encoding':'gb18030','order_column':order}}
    result=convert_dataset(cfg,tmp_path/'converted',progress=False)
    expected=[[10.123456789012345,2,100],[11,4,101]]
    if order:
        expected.reverse()
    np.testing.assert_array_equal(read_sequence_csv(result.parent/'0/0.csv'),expected)
    np.testing.assert_array_equal(read_sequence_csv(result.parent/'1/0.csv'),[[20,3,200]])
    assert '10.123456789012345' in (result.parent/'0/0.csv').read_text()
    assert load_yaml(result)['features'] == FEATURES


@pytest.mark.parametrize('problem', ['missing','extra','duplicate','noninteger','out_of_range','regex'])
def test_bad_flat_labels_and_unmatched_filenames_are_rejected(tmp_path,problem):
    kind='filename' if problem=='regex' else 'label_file'
    cfg,_=make_inputs(tmp_path/'raw',kind)
    if problem=='regex':
        (tmp_path/'raw/class0_sample0.csv').rename(tmp_path/'raw/unknown.csv')
    else:
        file=tmp_path/'raw/labels.csv'
        body=file.read_text(encoding='utf-8-sig')
        if problem=='missing': body=body.replace('class0_sample0.csv,0\n','')
        if problem=='extra': body+='absent.csv,0\n'
        if problem=='duplicate': body+='class0_sample0.csv,0\n'
        if problem=='noninteger': body=body.replace('class0_sample0.csv,0','class0_sample0.csv,unknown')
        if problem=='out_of_range': body=body.replace('class0_sample0.csv,0','class0_sample0.csv,2')
        file.write_text(body,encoding='utf-8')
    with pytest.raises(ValueError):
        convert_dataset(cfg,tmp_path/'converted',progress=False)
    assert not (tmp_path/'converted').exists()


@pytest.mark.parametrize('problem', ['conflict','missing_id','missing_column','duplicate_order','nan','non_numeric','empty'])
def test_invalid_table_does_not_publish_a_partial_dataset(tmp_path,problem):
    source=tmp_path/'raw'
    source.mkdir()
    header=['sample_id','label','pulse_index',*FEATURES]
    rows=[['a',0,0,10,2,100],['a',0,1,20,4,200]]
    if problem=='conflict': rows[1][1]=1
    if problem=='missing_id': rows[1][0]=''
    if problem=='missing_column': header[0]='other'
    if problem=='duplicate_order': rows[1][2]=0
    if problem=='nan': rows[1][3]='NaN'
    if problem=='non_numeric': rows[1][3]='invalid'
    if problem=='empty': rows=[]
    with (source/'samples.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.writer(stream);writer.writerow(header);writer.writerows(rows)
    cfg={'root':str(source),'num_classes':2,'input':{'format':'table','order_column':'pulse_index'}}
    with pytest.raises(ValueError):
        convert_dataset(cfg,tmp_path/'converted',progress=False)
    assert not (tmp_path/'converted').exists()
    assert not list(tmp_path.glob('.converted.converting-*'))


@pytest.mark.parametrize('bad', [{'format':'unknown'},{'filename_regex':'[','format':'filename'},
    {'filename_regex':r'class(\d+)','format':'filename'},{'format':'table','header':False},
    {'sample_id_columns':[]},{'label_offset':True},{'label_map':{'x':-1}},{'unexpected':1}])
def test_invalid_input_options_are_rejected(bad):
    with pytest.raises((ValueError,TypeError)):
        input_config(bad)


@pytest.mark.parametrize('kind',FORMATS)
def test_converted_dataset_is_portable_and_independent_of_originals(tmp_path,kind):
    cfg,_=make_inputs(tmp_path/'raw',kind)
    description=tmp_path/'raw/source.yaml'
    description.write_text(yaml.safe_dump({**cfg,'root':'.','root_relative_to_yaml':True}),encoding='utf-8')
    result=convert_dataset(description,tmp_path/'converted',progress=False)
    result.parent.rename(tmp_path/'relocated')
    (tmp_path/'raw').rename(tmp_path/'source_unavailable')
    pipeline=get_cfg(REPO_ROOT/'configs/pipeline/cnn_softmax_synthetic.yaml',
                     {'data':tmp_path/'relocated/data.yaml','num_classes':2,'dataset':{'sample_ratio':1.}})
    assert Path(pipeline['data_cfg']['root']) == tmp_path/'relocated'
    assert select_records(pipeline['data_cfg'])[1]['total_count'] == 12
    assert load_split_arrays(pipeline['data_cfg'],'train')[0].shape[2] == 3


@pytest.mark.parametrize('target', ['existing','nested','ancestor'])
def test_conversion_never_overwrites_or_merges_source_and_output(tmp_path,target):
    cfg,_=make_inputs(tmp_path/'raw','filename')
    output={'existing':tmp_path/'converted','nested':tmp_path/'raw/converted','ancestor':tmp_path}[target]
    if target=='existing':
        output.mkdir()
        (output/'keep.txt').write_text('keep')
    with pytest.raises((FileExistsError,ValueError)):
        convert_dataset(cfg,output,progress=False)
    if target=='existing':
        assert (output/'keep.txt').read_text() == 'keep'
    assert len(list((tmp_path/'raw').glob('*.csv'))) == 12


def test_cli_converts_yaml_from_another_working_directory(tmp_path):
    cfg,_=make_inputs(tmp_path/'raw','filename')
    description=tmp_path/'raw/source.yaml'
    description.write_text(yaml.safe_dump({**cfg,'root':'.','root_relative_to_yaml':True}),encoding='utf-8')
    result=subprocess.run([sys.executable,str(REPO_ROOT/'tools/convert_dataset.py'),'--data',str(description),
                           '--output',str(tmp_path/'converted'),'--no-progress'],cwd=tmp_path,
                          capture_output=True,text=True)
    assert result.returncode == 0,result.stdout+result.stderr
    assert (tmp_path/'converted/0/class0_sample0.csv').is_file()


def test_interrupted_conversion_can_retry_without_partial_output_or_source_changes(tmp_path):
    cfg,_=make_inputs(tmp_path/'raw','filename')
    original=(tmp_path/'raw/class0_sample0.csv').read_bytes()
    with patch('pulsecls.data.conversion.read_sequence_csv',
               side_effect=[np.zeros((20,3)),ValueError('invalid second sample')]):
        with pytest.raises(ValueError,match='invalid second sample'):
            convert_dataset(cfg,tmp_path/'converted',progress=False)
    assert not (tmp_path/'converted').exists()
    assert not list(tmp_path.glob('.converted.converting-*'))
    assert (tmp_path/'raw/class0_sample0.csv').read_bytes() == original
    assert convert_dataset(cfg,tmp_path/'converted',progress=False).is_file()


def test_filename_text_labels_are_mapped_explicitly(tmp_path):
    cfg,_=make_inputs(tmp_path/'raw','filename')
    cfg['input'].update(filename_regex=r'(?P<label>class\d+)_sample\d+\.csv',
                        label_map={'class0':0,'class1':1})
    result=convert_dataset(cfg,tmp_path/'converted',progress=False)
    assert read_sequence_csv(result.parent/'0/class0_sample0.csv')[0,0] == 3000
    assert read_sequence_csv(result.parent/'1/class1_sample0.csv')[0,0] == 3100


@pytest.mark.parametrize('kind',FORMATS)
def test_training_validation_resume_and_folder_inference_after_conversion(tmp_path,kind):
    cfg,_=make_inputs(tmp_path/'raw',kind)
    description=convert_dataset(cfg,tmp_path/'converted',progress=False)
    converted=converted_config(description)
    device=os.environ.get('DATA_INPUT_TEST_DEVICE','cpu')
    pipeline={'task':'classify','output_dir':str(tmp_path/'run'),'data_cfg':converted,
              'model_cfg':{'type':'cnn1d','in_channels':3,'hidden_channels':[4,8],
                           'stream_embedding_dim':8,'contrast_dim':4,'dropout':0.},
              'train':{'device':device,'epochs':1,'batch_size':4,'use_amp':False,
                       'visualize':False,'normalization':'minmax','early_stopping_patience':100},
              'augment':{'enabled':False},'inference':{'progress':False},'loss':{'ce':{'weight':1.}}}
    api=PulseCLS()
    result=api.train(pipeline=pipeline)
    assert result['best_epoch']==1 and result['validation_enabled']
    checkpoint=tmp_path/'run/last_model.pt'
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert saved['data_cfg']['input']['format']=='class_folders'
    assert min(float(state['step']) for state in saved['optimizer_state_dict']['state'].values())>0
    assert saved['normalization']['config']['method']=='minmax'
    membership=json.loads((tmp_path/'run/dataset_selection.json').read_text())
    assert (membership['train_count'],membership['val_count'])==(9,3)
    assert membership['input']['format']=='class_folders'
    assert all('#sample=' not in row['path'] for row in membership['sampled'])
    metrics=api.val(pipeline=pipeline,checkpoint=checkpoint,save_dir=tmp_path/'val')
    assert np.isfinite(metrics['macro_f1'])
    source=tmp_path/'test'
    source.mkdir()
    (source/'test.csv').write_text('CF_MHz,PW_us,PRI_us\n3100,11,501\n3101,12,502\n',encoding='utf-8')
    api.infer(pipeline=pipeline,checkpoint=checkpoint,source=source,save_dir=tmp_path/'predictions',
              inference={'batch_size':2,'progress':False})
    assert (tmp_path/'predictions/predictions.csv').is_file()
    api.train(pipeline=pipeline,output_dir=tmp_path/'resumed',resume=True,resume_checkpoint=checkpoint,epochs=1)
    resumed=torch.load(tmp_path/'resumed/last_model.pt',map_location='cpu',weights_only=False)
    assert resumed['epoch']==2 and resumed['normalization']==saved['normalization']
    assert min(float(state['step']) for state in resumed['optimizer_state_dict']['state'].values())>min(
        float(state['step']) for state in saved['optimizer_state_dict']['state'].values())
    assert all(torch.isfinite(v).all() for v in resumed['model_state_dict'].values())
