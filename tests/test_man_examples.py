import json,unittest
from em_skillopt.harness.man_examples import normalize_man_example


class ManExamples(unittest.TestCase):
    def test_only_plan_value_changes_preserving_prose_and_action_numbers(self):
        before='Human: Don\'t move the red cube.\nInput: {\'x\': 7}\nOutput: {\n "reasoning": "Preserve this wording.",\n "executable_plan": "[[50,42,17,6,61,36,1], [50,42,17,6,61,36,0]]"\n}'
        after=normalize_man_example(before)
        prefix,suffix=before.split('"executable_plan": ')
        self.assertTrue(after.startswith(prefix+'"executable_plan": '))
        self.assertTrue(after.endswith('\n}'))
        document=json.loads(after.split('Output: ')[1])
        self.assertEqual(document['reasoning'],'Preserve this wording.')
        self.assertEqual(document['executable_plan'],[{'action':[50,42,17,6,61,36,1]},{'action':[50,42,17,6,61,36,0]}])
        self.assertNotIn('language_plan',document)
        self.assertEqual(normalize_man_example(after),after)

    def test_extra_grouping_bracket_preserves_every_numeric_vector(self):
        value='[[1,2,3,4,5,6,0]], [7,8,9,10,11,12,1]]'
        result=json.loads(normalize_man_example(json.dumps({'executable_plan':value})))
        self.assertEqual(result['executable_plan'],[{'action':[1,2,3,4,5,6,0]},{'action':[7,8,9,10,11,12,1]}])
        with self.assertRaises(ValueError):
            normalize_man_example(json.dumps({'executable_plan':'[[1,2,3,4,5,6,0]], move_to()]'}))

    def test_already_object_and_empty_plans_are_supported(self):
        for value in [[],[{'action':[0,1,2,3,4,5,0]}]]:
            text=json.dumps({'executable_plan':value})
            self.assertEqual(json.loads(normalize_man_example(text))['executable_plan'],value)


if __name__=='__main__':unittest.main()
