"""Example decision query; the simulator defines state transitions itself."""
import argparse
import json
from decoder_decision import DecisionModel

p=argparse.ArgumentParser()
p.add_argument('--model',required=True,help='Text causal-LM checkpoint or HF repo')
p.add_argument('--device',default='auto')
args=p.parse_args()
model=DecisionModel.from_pretrained(args.model,device=args.device)
state={
    'resident': {'budget':120,'trust_in_neighbors':0.7,'preferences':['fairness','financial_stability']},
    'community': {'shared_project_cost':300,'other_residents_pledged':220},
    'situation': 'Residents are choosing whether to contribute to a community repair project.'}
questions={
    'participation': {'question':'Given the resident profile and current situation, which decision would this resident most likely make?',
        'options': {'contribute':'Contribute 40 units to the repair project.',
                    'wait':'Wait for more information before committing.',
                    'decline':'Decline to contribute.'}},
    'communication': {'question':'How would the resident most likely communicate their decision?',
        'options': {'public':'Tell the community publicly.',
                    'private':'Speak privately with the organizer.',
                    'silent':'Do not communicate yet.'}}}
print(json.dumps(model.decide(state,questions),ensure_ascii=False,indent=2))
