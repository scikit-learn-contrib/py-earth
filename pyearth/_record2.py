from ._util import ascii_table

class PruningPassRecord(object):
    def __init__(self, score_type, unpruned_terms, unpruned_score=None, unpruned_r2=None, record=None, selection=None):
        self.score_type = score_type
        self.unpruned_terms = unpruned_terms
        self.unpruned_score = unpruned_score
        self.unpruned_r2 = unpruned_r2
        if record is None:
            self.record = []
        else:
            self.record = record
        self.selection = selection
        
    def add(self, prune_set, score, r2):
        if len(prune_set) == 0:
            self.unpruned_score = score
            self.unpruned_r2 = r2
        else:
            self.record.append((prune_set, score, r2))
    
    def __reduce__(self):
        return (self.score_type, self.unpruned_terms, self.unpruned_score, self.unpruned_r2, 
                self.record, self.selection)
    
    def select(self, selection):
        self.selection = selection
    
    def __str__(self):
        result = ''
        result += 'Pruning Pass (%s)\n' % self.score_type
        result += 'Total terms: %d\tUnpruned Score: %.3f\tUnpruned R^2: %.3f\n' % (self.unpruned_terms, self.unpruned_score, self.unpruned_r2) , 
        header = ['iteration', 'pruned', 'terms', self.score_type, 'R^2']
        data = []
        for i, (prune_set, score, r2) in enumerate(self.record):
            data.append(i, ', '.join([str(p) for p in prune_set]),
                        self.unpruned_terms - len(prune_set), score, r2)
        result += ascii_table(header, data)
        result += '\nSelected iteration: ' + str(self.selection) + '\n'
    
# class ForwardPassRecord(object):
#     def __init__(self, score_type):
#         
#             
        