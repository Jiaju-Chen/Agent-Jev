import importlib.util
import unittest
from pathlib import Path

spec=importlib.util.spec_from_file_location('prepare_dataset',Path(__file__).resolve().parents[1]/'tools/prepare_dataset.py')
prepare=importlib.util.module_from_spec(spec);spec.loader.exec_module(prepare)


class DatasetTest(unittest.TestCase):
    def test_index_and_text_target_mapping(self):
        row=dict(context='state',choices=['stay','go'],label=1,person='p1')
        converted=prepare.convert(row,'context','unused','choices','label','index',question='Which?',group_field='person')
        self.assertEqual(converted['target'],'1')
        self.assertEqual(converted['options']['1'],'go')
        row['label']='stay'
        converted=prepare.convert(row,'context','unused','choices','label','text',question='Which?')
        self.assertEqual(converted['target'],'0')
        row['label']=-1
        with self.assertRaises(ValueError):
            prepare.convert(row,'context','unused','choices','label','index',question='Which?')

    def test_grouped_split_and_duplicate_input_split(self):
        rows=[dict(task_id=str(i//2),state=str(i),question='Q',options={'a':'A'},target='a') for i in range(6)]
        train,dev=prepare.split_groups(rows,.4,1)
        self.assertFalse({r['task_id'] for r in train}&{r['task_id'] for r in dev})
        unique=[dict(state=str(i),question='Q',options={'a':'A'},target='a') for i in range(3)]
        train,dev=prepare.split_groups(unique+[unique[0]],.4,2)
        self.assertFalse({r['state'] for r in train}&{r['state'] for r in dev})

    def test_csv_and_arc_style_options(self):
        self.assertEqual(prepare.options('["stay", "go"]'),{'0':'stay','1':'go'})
        self.assertEqual(prepare.options({'label':['A','B'],'text':['stay','go']}),{'A':'stay','B':'go'})
        with self.assertRaises(ValueError):
            prepare.options({'label':['A','A'],'text':['stay','go']})
