"""Equivalent logical samples across supported competition input layouts."""
import csv
import json
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
import yaml

from pulsecls import PulseCLS
from pulsecls.cfg import REPO_ROOT, get_cfg
from pulsecls.data.dataset import load_sample, load_split_arrays, read_sequence_csv
from pulsecls.data.file_index import load_inventory, record_at
from pulsecls.data.selection import select_records
from pulsecls.data.sources import TableSample, input_config


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


@pytest.mark.parametrize('kind', FORMATS)
def test_layouts_preserve_samples_labels_and_disjoint_reproducible_splits(tmp_path, kind):
    cfg, expected = make_inputs(tmp_path, kind)
    plan = select_records(cfg)
    parts, info = plan
    assert (info['total_count'],info['train_count'],info['val_count']) == (12,9,3)
    assert not set(parts['train']) & set(parts['val'])
    assert select_records(cfg)[0] == parts
    actual = [(label,read_sequence_csv(ref)) for ref,label in parts['all']]
    assert sorted((label,len(raw),float(raw[0,0])) for label,raw in actual) == sorted(
        (label,len(raw),float(raw[0,0])) for label,raw in expected)
    by_start = {float(raw[0,0]):raw for _,raw in expected}
    for label, raw in actual:
        np.testing.assert_allclose(raw,by_start[float(raw[0,0])],atol=1e-10)
    with patch('pulsecls.data.dataset.load_sample', wraps=load_sample) as reader:
        train, train_y, train_paths = load_split_arrays(cfg,'train',selection_plan=plan)
        val, val_y, val_paths = load_split_arrays(cfg,'val',selection_plan=plan)
        assert reader.call_count == 12
    assert train.shape == (9,32,3) and val.shape == (3,32,3)
    assert not set(train_paths) & set(val_paths)
    with patch('pulsecls.data.dataset.load_sample', side_effect=AssertionError('CSV reread despite valid cache')):
        cached = load_split_arrays(cfg,'train')
    np.testing.assert_array_equal(cached[0],train)
    np.testing.assert_array_equal(cached[1],train_y)
    assert cached[2] == train_paths
    assert set(train_y) == set(val_y) == {0,1}
    # Existing files stay in place. A table does not create one CSV per logical sample.
    assert len(list(tmp_path.glob('*.csv'))) == (1 if kind=='table' else 13 if kind=='label_file' else 12)


@pytest.mark.parametrize('delimiter,header', [(',',True),('\t',True),('whitespace',False)])
def test_label_table_csv_tsv_and_headerless_txt(tmp_path, delimiter, header):
    cfg, _ = make_inputs(tmp_path,'label_file')
    (tmp_path/'labels.csv').unlink()
    lines = ['path,target'] if header else []
    lines += [f'class{label}_sample{i}.csv,{label+1}' for label in range(2) for i in range(6)]
    separator = ' ' if delimiter=='whitespace' else delimiter
    (tmp_path/'labels.txt').write_text('\n'.join(line.replace(',',separator) for line in lines),encoding='utf-8-sig')
    cfg['input'].update(labels_file='labels.txt',delimiter=delimiter,header=header,
                        filename_column='path',label_column='target',label_offset=-1)
    groups,_,_ = load_inventory(tmp_path,cfg)
    assert [(g['label'],g['count']) for g in groups] == [(0,6),(1,6)]


def test_filename_mapping_and_explicit_configuration_priority(tmp_path):
    cfg,_ = make_inputs(tmp_path,'filename')
    cfg['input']['filename_regex'] = r'(?P<label>class\d+)_sample\d+\.csv'
    cfg['input']['label_map'] = {'class0':0,'class1':1}
    assert select_records(cfg)[1]['sampled_count'] == 12
    pipeline = {'task':'classify','output_dir':str(tmp_path/'run'),'data_cfg':cfg,'model_cfg':{'type':'cnn1d'}}
    resolved = get_cfg(pipeline,{'data_input':{'format':'filename','filename_regex':r'class(?P<label>\d+)_sample\d+\.csv'}})
    assert resolved['data_cfg']['input']['label_map'] == {}
    assert resolved['data_cfg']['input']['format'] == 'filename'
    assert get_cfg(resolved,{'data_cfg.input.format':'class_folders'})['data_cfg']['input']['format']=='class_folders'
    description = tmp_path/'data.yaml'
    description.write_text(yaml.safe_dump({**cfg,'root':'.','root_relative_to_yaml':True}),encoding='utf-8')
    loaded = get_cfg(REPO_ROOT/'configs/pipeline/cnn_softmax_synthetic.yaml',{'data':description,'num_classes':2,'dataset':{'sample_ratio':1.}})
    assert loaded['data_cfg']['input']['format']=='filename'
    assert select_records(loaded['data_cfg'])[1]['total_count']==12


@pytest.mark.parametrize('kind', FORMATS)
def test_ready_to_edit_data_templates_load(kind):
    template=REPO_ROOT/f'configs/data/{kind}.yaml'
    cfg=get_cfg(REPO_ROOT/'configs/pipeline/cnn_softmax_synthetic.yaml',{'data':template})
    assert cfg['data_cfg']['input']['format']==kind
    assert Path(cfg['data_cfg']['root'])==(REPO_ROOT/'datasets/competition').resolve()


def test_raw_csv_changes_invalidate_arrays_and_input_changes_reject_old_plan(tmp_path):
    cfg,_=make_inputs(tmp_path,'filename')
    original=select_records(cfg)
    before=load_split_arrays(cfg,'all')[0]
    path=tmp_path/'class0_sample0.csv'
    path.write_text(path.read_text().replace('3.000000000000000000e+03','9.000000000000000000e+03'),encoding='utf-8')
    after=load_split_arrays(cfg,'all')[0]
    assert not np.array_equal(before,after)
    cfg['input']['filename_regex']=r'class(?P<label>\d+)_sample\d+\.csv'
    with pytest.raises(ValueError,match='input format/options'):
        load_split_arrays(cfg,'train',selection_plan=original)


def test_table_dataset_visualization_uses_original_logical_sequences(tmp_path):
    from tools.visualize_dataset import visualize_dataset
    cfg,_=make_inputs(tmp_path/'data','table')
    cfg.update(root='.',root_relative_to_yaml=True)
    description=tmp_path/'data/data.yaml'
    description.write_text(yaml.safe_dump(cfg),encoding='utf-8')
    result=visualize_dataset(description,tmp_path/'plots',sequences_per_class=2,max_sequences=4,
                             plot_classes=1,examples_per_class=1,pulses_per_sequence=2,
                             max_plot_points=5,bins=4,dpi=40,formats=('png',),progress=False)
    assert (tmp_path/'plots/index.html').is_file()


@pytest.mark.parametrize('kind', FORMATS)
def test_index_reuse_and_invalidation_after_source_or_label_change(tmp_path, kind):
    cfg,_ = make_inputs(tmp_path,kind)
    first = select_records(cfg)
    reader = '_table_records' if kind=='table' else '_flat_records'
    with patch('pulsecls.data.sources.'+reader,side_effect=AssertionError('Unnecessary index rebuild')):
        assert select_records(cfg)[0] == first[0]
    data1,_,_ = load_split_arrays(cfg,'all')
    if kind == 'label_file':
        label_path = tmp_path/'labels.csv'
        label_path.write_text(label_path.read_text(encoding='utf-8-sig').replace('class0_sample0.csv,0','class0_sample0.csv,1'),encoding='utf-8')
    elif kind == 'filename':
        file = tmp_path/'class0_sample0.csv'
        file.rename(tmp_path/'class1_sample999.csv')
    else:
        table=tmp_path/'samples.csv'
        with table.open('a',newline='',encoding='utf-8') as stream:
            csv.writer(stream).writerow(['new',0,0,9000,12,800])
    second = select_records(cfg)
    assert second[1]['source_fingerprint'] != first[1]['source_fingerprint']
    data2,_,_ = load_split_arrays(cfg,'all')
    assert len(data2) == len(data1)+(kind=='table')


def test_table_composite_ids_multiline_ids_and_preserved_file_order(tmp_path):
    path=tmp_path/'samples.csv'
    with path.open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.writer(stream)
        writer.writerow(['sample_id','label',*FEATURES])
        writer.writerows([['shared\nID',0,10,2,100],['shared\nID',1,20,3,200],['shared\nID',0,11,4,101]])
    cfg={'num_classes':2,'input':{'format':'table','sample_id_columns':['label','sample_id']}}
    inventory,_,_=load_inventory(tmp_path,cfg)
    ref,label=record_at(inventory,0)
    assert isinstance(ref,TableSample) and label==0
    np.testing.assert_array_equal(read_sequence_csv(ref),[[10,2,100],[11,4,101]])
    np.testing.assert_array_equal(load_sample(ref,4,length_policy={'short':'repeat','long':'center'}),
                                  [[10,2,100],[11,4,101],[10,2,100],[11,4,101]])


@pytest.mark.parametrize('problem', ['missing','extra','duplicate','noninteger','out_of_range','regex'])
def test_bad_flat_labels_and_unmatched_filenames_are_rejected(tmp_path, problem):
    kind='filename' if problem=='regex' else 'label_file'
    cfg,_=make_inputs(tmp_path,kind)
    if problem=='regex':
        (tmp_path/'class0_sample0.csv').rename(tmp_path/'unknown.csv')
    else:
        file=tmp_path/'labels.csv'
        body=file.read_text(encoding='utf-8-sig')
        if problem=='missing': body=body.replace('class0_sample0.csv,0\n','')
        if problem=='extra': body+='absent.csv,0\n'
        if problem=='duplicate': body+='class0_sample0.csv,0\n'
        if problem=='noninteger': body=body.replace('class0_sample0.csv,0','class0_sample0.csv,unknown')
        if problem=='out_of_range': body=body.replace('class0_sample0.csv,0','class0_sample0.csv,2')
        file.write_text(body,encoding='utf-8')
    with pytest.raises(ValueError):
        select_records(cfg)


@pytest.mark.parametrize('problem', ['conflict','missing_id','missing_column','duplicate_order','nan'])
def test_invalid_table_samples_are_rejected(tmp_path, problem):
    header=['sample_id','label','pulse_index',*FEATURES]
    rows=[['a',0,0,10,2,100],['a',0,1,20,4,200]]
    if problem=='conflict': rows[1][1]=1
    if problem=='missing_id': rows[1][0]=''
    if problem=='missing_column': header[0]='other'
    if problem=='duplicate_order': rows[1][2]=0
    if problem=='nan': rows[1][3]='NaN'
    with (tmp_path/'samples.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.writer(stream);writer.writerow(header);writer.writerows(rows)
    cfg={'num_classes':2,'input':{'format':'table','order_column':'pulse_index'}}
    with pytest.raises(ValueError):
        inventory,_,_=load_inventory(tmp_path,cfg)
        read_sequence_csv(record_at(inventory,0)[0])


@pytest.mark.parametrize('bad', [{'format':'unknown'},{'filename_regex':'[' ,'format':'filename'},
    {'filename_regex':r'class(\d+)','format':'filename'},{'format':'table','header':False},
    {'sample_id_columns':[]},{'label_offset':True},{'label_map':{'x':-1}},{'unexpected':1}])
def test_invalid_input_options_are_rejected(bad):
    with pytest.raises((ValueError,TypeError)):
        input_config(bad)


@pytest.mark.parametrize('kind', FORMATS)
def test_training_validation_resume_and_folder_inference(tmp_path, kind):
    cfg,_=make_inputs(tmp_path/'data',kind)
    device=os.environ.get('DATA_INPUT_TEST_DEVICE','cpu')
    epochs=int(os.environ.get('DATA_INPUT_TEST_EPOCHS','1'))
    pipeline={'task':'classify','output_dir':str(tmp_path/'run'),'data_cfg':cfg,
              'model_cfg':{'type':'cnn1d','in_channels':3,'hidden_channels':[4,8],
                           'stream_embedding_dim':8,'contrast_dim':4,'dropout':0.},
              'train':{'device':device,'epochs':epochs,'batch_size':4,'use_amp':False,
                       'visualize':False,'normalization':'minmax','early_stopping_patience':100},
              'augment':{'enabled':False},'inference':{'progress':False},'loss':{'ce':{'weight':1.}}}
    api=PulseCLS()
    result=api.train(pipeline=pipeline)
    assert 1<=result['best_epoch']<=epochs and result['validation_enabled']
    assert str(api.trainer.device)==device
    checkpoint=tmp_path/'run/last_model.pt'
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert saved['epoch']==epochs
    assert min(float(state['step']) for state in saved['optimizer_state_dict']['state'].values())>0
    assert saved['data_cfg']['input']['format']==kind
    assert saved['normalization']['config']['method']=='minmax'
    membership=json.loads((tmp_path/'run/dataset_selection.json').read_text())
    assert (membership['train_count'],membership['val_count'])==(9,3)
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
    assert resumed['epoch']==epochs+1 and resumed['normalization']==saved['normalization']
    assert min(float(state['step']) for state in resumed['optimizer_state_dict']['state'].values())>min(
        float(state['step']) for state in saved['optimizer_state_dict']['state'].values())
    assert all(torch.isfinite(v).all() for v in resumed['model_state_dict'].values())
