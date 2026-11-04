import json
import inspect
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from bert_baseline.data import Example, read_xml, split_train_validation, ensure_disjoint, classification_metrics

ROOT = Path(__file__).resolve().parents[1]

class DataTests(unittest.TestCase):
    def test_actual_downloaded_xml(self):
        database = ROOT / 'Database'
        semeval = database / 'SemEval2014'
        files, missing = [], []
        for name in ('Laptop_Train_v2.xml', 'Restaurants_Train_v2.xml'):
            matches = list(semeval.rglob(name))
            if len(matches) == 1:
                files.append(matches[0])
            else:
                missing.append(f'{semeval}/**/{name} (found {len(matches)})')
        expected = [semeval / 'ABSA_Gold_TestData' / name for name in
                    ('Laptops_Test_Gold.xml', 'Restaurants_Test_Gold.xml')]
        expected += [database / 'MAMS/data/MAMS-ATSA/raw' / f'{split}.xml'
                     for split in ('train', 'val', 'test')]
        for path in expected:
            if path.is_file():
                files.append(path)
            else:
                missing.append(str(path))
        self.assertFalse(missing,
                         '\nDataset files missing or ambiguous:\n' + '\n'.join(missing) +
                         '\nDownload/copy the datasets into the Database directory described in README.md.' +
                         '\nModel-only offline tests: python -m unittest discover -s tests -p "test_enhanced.py" -v')
        for path in files:
            examples, stats = read_xml(path)
            self.assertTrue(examples)
            self.assertNotIn('conflict', {x.label for x in examples})
            self.assertEqual(stats['examples'], len(examples))
            all_examples, all_stats = read_xml(path, include_conflict=True)
            self.assertEqual(len(all_examples), all_stats['aspects_total'])

    def test_grouped_split_reproducible(self):
        rows = [Example(str(i), f'sentence {i//2}', 'sentence', 0, 8, 'positive') for i in range(20)]
        a,b = split_train_validation(rows, seed=42)
        self.assertEqual((a,b), split_train_validation(rows, seed=42))
        self.assertEqual(len(a)+len(b), len(rows))
        ensure_disjoint(train=a, validation=b)
        with self.assertRaises(ValueError):
            ensure_disjoint(train=a, validation=a)

    def test_metrics_known_confusion(self):
        result = classification_metrics([0,0,1,2], [0,1,1,2], ['negative','neutral','positive'])
        self.assertEqual(result['accuracy'], 0.75)
        self.assertAlmostEqual(result['macro_f1'], 7/9)
        self.assertEqual(result['confusion_matrix'], [[1,1,0],[0,1,0],[0,0,1]])
        with self.assertRaises(ValueError):
            classification_metrics([], [], ['positive'])

    def test_bad_offsets_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'bad.xml'
            path.write_text('<sentences><sentence id="x"><text>good food</text><aspectTerms>'
                            '<aspectTerm term="food" from="0" to="4" polarity="positive"/>'
                            '</aspectTerms></sentence></sentences>')
            with self.assertRaises(ValueError):
                read_xml(path)

class ModelTests(unittest.TestCase):
    def setUp(self):
        import torch
        from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        vocab = ['[PAD]','[UNK]','[CLS]','[SEP]','[MASK]', 'the','food','was','good','bad','service',
                 'okay','but','slow','.','filler','target']
        (self.directory/'vocab.txt').write_text('\n'.join(vocab)+'\n')
        vocab_key = 'vocab' if 'vocab' in inspect.signature(BertTokenizerFast).parameters else 'vocab_file'
        self.tokenizer = BertTokenizerFast(**{vocab_key: str(self.directory/'vocab.txt')}, do_lower_case=True)
        # Verify this constructor actually loaded the intended vocabulary.
        self.assertNotEqual(self.tokenizer.convert_tokens_to_ids('target'), self.tokenizer.unk_token_id)
        self.model = BertForSequenceClassification(BertConfig(
            vocab_size=len(vocab), hidden_size=24, num_hidden_layers=1, num_attention_heads=2,
            intermediate_size=32, max_position_embeddings=128, num_labels=3,
            id2label={0:'negative',1:'neutral',2:'positive'},
            label2id={'negative':0,'neutral':1,'positive':2}))

    def test_target_preserving_truncation_and_length_guard(self):
        from bert_baseline.model import encode_example
        text = 'filler '*80+'target'
        row = Example('x',text,'target',text.index('target'),len(text),'positive')
        encoded = encode_example(self.tokenizer,row,16)
        tokens = self.tokenizer.convert_ids_to_tokens(encoded['input_ids'])
        first_sep = tokens.index('[SEP]')
        self.assertIn('target', tokens[:first_sep])
        self.assertIn('target', tokens[first_sep+1:])
        self.assertLessEqual(len(tokens),16)
        with self.assertRaises(ValueError):
            encode_example(self.tokenizer,row,3)

    def test_forward_backward_save_reload(self):
        import torch
        from transformers import BertForSequenceClassification, DataCollatorWithPadding
        from bert_baseline.model import AspectDataset
        row=Example('x','the food was good','food',4,8,'positive')
        data=AspectDataset([row],self.tokenizer,['negative','neutral','positive'],32)
        batch=DataCollatorWithPadding(self.tokenizer)([data[0]])
        self.model.train()
        output=self.model(**batch)
        self.assertEqual(tuple(output.logits.shape),(1,3))
        self.assertTrue(torch.isfinite(output.loss).item())
        output.loss.backward()
        self.assertIsNotNone(self.model.bert.embeddings.word_embeddings.weight.grad)
        self.model.eval()
        with torch.no_grad():
            before=self.model(**batch).logits
        self.model.save_pretrained(self.directory/'saved')
        restored=BertForSequenceClassification.from_pretrained(self.directory/'saved').eval()
        with torch.no_grad():
            after=restored(**batch).logits
        torch.testing.assert_close(before,after)

    def test_cli_train_evaluate_predict(self):
        checkpoint=self.directory/'initial'
        self.model.save_pretrained(checkpoint)
        self.tokenizer.save_pretrained(checkpoint)
        def fixture(split, suffix):
            path=self.directory/f'{split}.xml'
            sentences=[]
            for i,(word,label) in enumerate([('good','positive'),('bad','negative'),('okay','neutral')]):
                text=f'the food was {word} {suffix}'
                sentences.append(f'<sentence id="{split}{i}"><text>{text}</text><aspectTerms>'
                                 f'<aspectTerm term="food" from="4" to="8" polarity="{label}"/>'
                                 '</aspectTerms></sentence>')
            if split == 'train':
                sentences.append('<sentence id="overlap"><text>the food was good slow</text>'
                                 '<aspectTerms><aspectTerm term="food" from="4" to="8" '
                                 'polarity="positive"/></aspectTerms></sentence>')
            path.write_text('<sentences>'+''.join(sentences)+'</sentences>')
            return path
        paths=[fixture('train','.'), fixture('val','but'), fixture('test','slow')]
        def run(*args):
            result=subprocess.run([sys.executable,'-m','bert_baseline',*map(str,args)],
                                  cwd=ROOT,capture_output=True,text=True,timeout=120)
            self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)
            return result.stdout
        output=self.directory/'run'
        run('train','--model',checkpoint,'--train',paths[0],'--validation',paths[1],
            '--test',paths[2],'--output',output,'--epochs',1,'--batch-size',3,'--max-length',32,'--device','cpu')
        config=json.loads((output/'run_config.json').read_text())
        self.assertEqual(config['excluded_test_overlap']['train_examples'],1)
        self.assertEqual(config['train_examples'],3)
        metrics=json.loads((output/'test_metrics.json').read_text())
        self.assertEqual(metrics['examples'],3)
        self.assertEqual(len((output/'predictions.jsonl').read_text().splitlines()),3)
        run('evaluate','--checkpoint',output/'best','--test',paths[2],'--output',self.directory/'evaluation','--device','cpu')
        prediction=json.loads(run('predict','--checkpoint',output/'best','--text','the food was good',
                                  '--aspect','food','--device','cpu'))
        self.assertIn(prediction['prediction'],['negative','neutral','positive'])
        self.assertAlmostEqual(sum(prediction['probabilities'].values()),1,places=5)

if __name__=='__main__':
    unittest.main()
