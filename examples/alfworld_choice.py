"""One ALFWorld-style decision; environment installation is unnecessary here."""
import argparse
import json
from decoder_decision import DecisionModel


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',required=True)
    p.add_argument('--device',default='auto')
    p.add_argument('--row',help='Optional JSON file with task/observation/history/legal_actions')
    args=p.parse_args()
    row=json.loads(open(args.row).read()) if args.row else dict(
        task='put a clean apple in the fridge',
        observation='You are in a kitchen. You see a sink, a countertop, and a fridge.',
        history=[],legal_actions=['go to sinkbasin 1','go to countertop 1','go to fridge 1','inventory','look'])
    model=DecisionModel.from_pretrained(args.model,device=args.device)
    print(json.dumps(model.decide_alfworld(row,seed=0),indent=2))


if __name__=='__main__':
    main()
