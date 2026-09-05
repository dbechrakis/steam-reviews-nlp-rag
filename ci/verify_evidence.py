"""Validate committed evidence without pretending to retrain external-data models."""
from pathlib import Path
import ast, csv, json, math
ROOT=Path(__file__).resolve().parents[1]
for p in ROOT.rglob('*.py'):
    if not any(x in p.parts for x in ['.git','.venv','node_modules']): ast.parse(p.read_text())
for p in ROOT.rglob('*.ipynb'):
    n=json.loads(p.read_text()); assert n['nbformat']==4
    for c in n['cells']:
        assert c['cell_type'] in ['code','markdown','raw']
        assert not any(o.get('output_type')=='error' for o in c.get('outputs',[])), str(p)
def table(path):
    with (ROOT/path).open(newline='',encoding='latin1' if str(path).startswith('data/') else 'utf-8') as f:return list(csv.DictReader(f))
def close(a,b,tol=1e-6): assert math.isclose(float(a),float(b),abs_tol=tol,rel_tol=tol),(a,b)
rows={r['split']:r for r in table('outputs/tables/full_data_and_sample_summary.csv')}
assert int(rows['train']['reviews'])+int(rows['test']['reviews'])==int(rows['modelling sample']['reviews'])
for r in table('outputs/tables/task1_model_comparison.csv'):
    for k in ['accuracy','macro_f1','negative_f1']:assert 0<=float(r[k])<=1
for path in ['outputs/tables/rag_model_comparison.csv','outputs/tables/rag_groundedness.csv']:
    for r in table(path):assert int(r['grounded'])+int(r['hallucinated'])==int(r['games_mentioned'])

print('Committed evidence and syntax checks passed; see VALIDATION.md for rerun scope.')
