import json
from pathlib import Path
import tempfile
import unittest
from run_textqa import extract_choices, match_answer, read_items, run_one, main, score_answer, metrics, LLMError


class TextQATest(unittest.TestCase):
    def test_exact_scoring(self):
        for value in ('c,a', 'AC', '答案：Ａ，Ｃ。', 'Answer: A,C', r'\boxed{A,C}', '**A,C**'):
            self.assertTrue(match_answer(value, 'A,C'), value)
        for value in ('A', 'A,B,C', 'A,A,C', '', 'Because A and C are correct', 'CAT'):
            self.assertFalse(match_answer(value, 'A,C'), value)
        self.assertTrue(match_answer('A,D,G,K,M', 'M,K,G,D,A'))

    def test_partial_scoring(self):
        for prediction, expected in (('A,C,F', 1.0), ('F,A,C', 1.0),
                                     ('A', 0.5), ('A,F', 0.5),
                                     ('A,B', 0.0), ('A,B,C,F', 0.0),
                                     ('', 0.0), ('A,A', 0.0), ('Explanation: A', 0.0)):
            self.assertEqual(score_answer(prediction, 'A,C,F'), expected, prediction)
        self.assertEqual(score_answer('B', 'B'), 1.0)
        self.assertEqual(score_answer('A', 'B'), 0.0)
        self.assertEqual(score_answer('A', ''), 0.0)
        records = [dict(score=score, correct=score == 1, error=None)
                   for score in (1.0, 0.5, 0.0)]
        summary = metrics(records)
        self.assertEqual(summary['total_score'], 1.5)
        self.assertEqual(summary['score_rate'], 0.5)
        self.assertEqual(summary['partial'], 1)
        self.assertEqual(summary['correct'], 1)

    def test_record_scores_and_errors(self):
        item = dict(id='x', question='Question', answer='A,C', type='multiple_choice')
        record = run_one(None, item, 0, prediction='A')
        self.assertEqual(record['score'], 0.5)
        self.assertFalse(record['correct'])
        class FailedClient:
            def chat(self, messages):
                raise LLMError('test failure')
        failed = run_one(FailedClient(), item, 0)
        self.assertEqual(failed['score'], 0.0)
        self.assertEqual(metrics([failed])['errors'], 1)

    def test_dataset_and_prompt(self):
        path = Path(__file__).with_name('textqa1.0.jsonl')
        for language in ('ch', 'en'):
            items = read_items(path, language)
            self.assertEqual(len(items), 80)
            class FakeClient:
                def chat(self, messages):
                    self.messages = messages
                    return {'content': 'C,B'}
            client = FakeClient()
            record = run_one(client, items[0], 0, language)
            self.assertTrue(record['correct'])
            self.assertEqual(client.messages[1]['content'], items[0]['question_' + language])
            self.assertEqual(len(client.messages), 2)

    def test_offline_and_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / 'input.jsonl'
            data.write_text(json.dumps(dict(id='x', question_ch='题目', question_en='Question', answer='A,C', field='Design'))+'\n')
            pred = root / 'pred.jsonl'
            pred.write_text('{"id":"x","answer":"C,A"}\n')
            self.assertEqual(main(['--input', str(data), '--predictions', str(pred), '--output', str(root/'results.jsonl')]), 0)
            summary = json.loads((root/'summary.json').read_text())
            self.assertEqual(summary['accuracy'], 1)
            self.assertEqual(summary['by_field']['Design']['correct'], 1)
            pred.write_text('{"id":"x","answer":"A"}\n')
            main(['--input', str(data), '--predictions', str(pred), '--output', str(root/'results.jsonl')])
            summary = json.loads((root/'summary.json').read_text())
            self.assertEqual(summary['score_rate'], 0.5)
            self.assertEqual(summary['accuracy'], 0)
            self.assertEqual(summary['by_field']['Design']['total_score'], 0.5)
            self.assertEqual(summary['by_material_category']['unknown']['partial'], 1)
            data.write_text('{"question_ch":"", "answer":"A"}\n')
            with self.assertRaises(ValueError):
                read_items(data)


if __name__ == '__main__':
    unittest.main()
