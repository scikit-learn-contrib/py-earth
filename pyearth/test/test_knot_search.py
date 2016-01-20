from pyearth._knot_search import OutcomeDependentData, KnotSearchWorkingData, \
    PredictorDependentData, KnotSearchReadOnlyData, KnotSearchData, knot_search
from nose.tools import assert_true, assert_equal
import numpy as np
from numpy.testing.utils import assert_almost_equal, assert_array_equal
from scipy.linalg import qr

def test_outcome_dependent_data():
    np.random.seed(10)
    m = 1000
    max_terms = 100
    y = np.random.normal(size=m)
    w = np.random.normal(size=m) ** 2
    data = OutcomeDependentData.alloc(y, w, m, max_terms)
    
    # Test updating
    B = np.empty(shape=(m,max_terms))
    for k in range(max_terms):
        b = np.random.normal(size=m)
        B[:,k] = b
        code = data.update(b, 1e-16)
        assert_equal(code, 0)
        assert_almost_equal(np.dot(data.Q_t[:k+1,:], np.transpose(data.Q_t[:k+1,:])),
                            np.eye(k+1))
    assert_equal(data.update(b, 1e-16), -1)
    
    # Test downdating
    q = np.array(data.Q_t).copy()
    theta = np.array(data.theta[:max_terms]).copy()
    data.downdate()
    data.update(b, 1e-16)
    assert_almost_equal(q, np.array(data.Q_t))
    assert_almost_equal(theta, np.array(data.theta[:max_terms]))
    assert_almost_equal(np.array(data.theta[:max_terms]), np.dot(data.Q_t, w * y))
    wB = B * w[:,None]
    Q, _ = qr(wB, pivoting=False, mode='economic')
    assert_almost_equal(np.abs(np.dot(data.Q_t, Q)), np.eye(max_terms))
    
    # Test that reweighting works
    assert_equal(data.k, max_terms)
    w2 = np.random.normal(size=m) ** 2
    data.reweight(w2, B, max_terms, 1e-16)
    assert_equal(data.k, max_terms)
    w2B = B * w2[:,None]
    Q2, _ = qr(w2B, pivoting=False, mode='economic')
    assert_almost_equal(np.abs(np.dot(data.Q_t, Q2)), np.eye(max_terms))
    assert_almost_equal(np.array(data.theta[:max_terms]), np.dot(data.Q_t, w2 * y))

def test_knot_candidates():
    np.random.seed(10)
    m = 1000
    x = np.random.normal(size=m)
    p = np.random.normal(size=m)
    p[np.random.binomial(p=.1, n=1,size=m)==1] = 0.
    x[np.random.binomial(p=.1, n=1,size=m)==1] = 0.
    predictor = PredictorDependentData.alloc(x)
    candidates, candidates_idx = predictor.knot_candidates(p, 5, 10, 0, 0)
    assert_array_equal(candidates, x[candidates_idx])
    assert_equal(len(candidates), len(set(candidates)))
    print candidates, np.sum(x==0)
#     print candidates_idx
    

def slow_knot_search(p, x, B, candidates, outcomes):
    # Brute force, utterly un-optimized knot search with no fast update.  Use only for
    # testing the actual knot search function.  This version allows for multiple outcome
    # columns.
    best_e = float('inf')
    best_k = 0
    best_knot = float('inf')
    for k, knot in enumerate(candidates):
        # Formulate the linear system for this candidate
        X = np.concatenate([B, (p * np.maximum(x-knot, 0.0))[:,None]], axis=1)

        # Solve the system for each y and w
        e_squared = 0.0
        for y, w in outcomes:
            # Solve the system
            beta = np.linalg.lstsq(w[:,None] * X, w * y)[0]

            # Compute the error
            r = w * (y - np.dot(X, beta))
            e_squared += np.dot(r, r)
        # Compute loss
        e = np.sqrt(e_squared)
    
        # Compare to the best error
        if e < best_e:
            best_e = e
            best_k = k
            best_knot = knot
    return best_knot, best_k, best_e

def generate_problem(m, q, r, n_outcomes, shared_weight):
    # Generate some problem data
    x = np.random.normal(size=m)
    B = np.random.normal(size=(m,q))
    p = B[:, 1]
    knot = x[int(m/2)]
    candidates = np.array(sorted([knot] + \
                    list(x[np.random.randint(low=0, high=m, size=r-1)])
                    ))[::-1]

    # These data need to be generated for each outcome
    outcomes = []
    if shared_weight:
        w = np.random.normal(size=m) ** 2
#         w = w * 0. + 1.
    for _ in range(n_outcomes):
        beta = np.random.normal(size=q+1)
        y = np.dot(np.concatenate([B, (p * np.maximum(x-knot, 0.0))[:,None]], axis=1), beta) + \
            0.01 * np.random.normal(size=m)
        if not shared_weight:
            w = np.random.normal(size=m) ** 2
#             w = w * 0. + 1.
        outcomes.append((y, w))

    return x, B, p, knot, candidates, outcomes


def form_inputs(x, B, p, knot, candidates, outcomes):
    # Formulate the inputs for the fast version
    m, q = B.shape
    max_terms = q + 2
    outcomes_ = []
    workings = []
    for y, w in outcomes:
        working = KnotSearchWorkingData.alloc(max_terms)
        workings.append(working)
        outcome = OutcomeDependentData.alloc(y, w, m, max_terms)
        outcome.reweight(w, B, q, 1e-16)
        assert_equal(outcome.k, q)
        outcomes_.append(outcome)
    predictor = PredictorDependentData.alloc(x)
    constant = KnotSearchReadOnlyData(predictor, outcomes_)
    return KnotSearchData(constant, workings, q)

def test_knot_search():
    seed = 10
    np.random.seed(seed)
    m = 100
    q = 5
    r = 10
    n_outcomes = 3
    
    # Generate some problem data
    x, B, p, knot, candidates, outcomes = generate_problem(m, q,  r, n_outcomes, False)

    # Formulate the inputs for the fast version
    data = form_inputs(x, B, p, knot, candidates, outcomes)
    
    # Get the answer using the slow version
    best_knot, best_k, best_e = slow_knot_search(p, x, B, candidates, outcomes)
    
    # Test the test
    assert_almost_equal(best_knot, knot)
    assert_equal(r, len(candidates))
    assert_equal(m, B.shape[0])
    assert_equal(q, B.shape[1])
    assert_equal(len(outcomes), n_outcomes)
    
    # Run fast knot search and compare results to slow knot search
    fast_best_knot, fast_best_k, fast_best_e = knot_search(data, candidates, 
                                                           p, q, m, r, 
                                                           len(outcomes))
    assert_almost_equal(fast_best_knot, best_knot)
    assert_equal(candidates[fast_best_k], candidates[best_k])
    assert_almost_equal(fast_best_e, best_e)
    
    
    



