"""One runnable check for experiment selection and candidate trust boundaries."""
from autoresearch import candidate, score

assert candidate('{"depth":1,"instruction":"Read imports before editing."}')['depth']==1
try:
    candidate('{"depth":2,"instruction":"Repair PACKAGE-MANIFEST first."}', ['package-manifest'])
except ValueError:
    pass
else:
    raise AssertionError('Task-specific proposal accepted')
for text in ['{"depth":3,"instruction":"x"}', '{"depth":1,"instruction":1}',
             '{"depth":1,"instruction":"x","checks":"pass"}']:
    try:
        candidate(text)
    except ValueError:
        pass
    else:
        raise AssertionError('Invalid proposal accepted')
good={'tasks':[{'passed':True,'output_tokens':20,'elapsed_s':2}]}
bad={'tasks':[{'passed':False,'output_tokens':1,'elapsed_s':1}]}
assert score(good)>score(bad)
assert score(good)>score({'tasks':[{'passed':True,'output_tokens':30,'elapsed_s':1}]})
print('PASS: quality-first selection and bounded proposal schema')
