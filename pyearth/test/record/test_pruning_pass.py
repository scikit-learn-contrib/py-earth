from pyearth._record import PruningPassRecord, PruningPassIteration
from pyearth._util import gcv
from ..testing_utils import assert_list_almost_equal


num_samples = 1000
num_variables = 10
penalty = 3.0
sst = 100.0
record = PruningPassRecord(num_samples, num_variables,
                           penalty, sst, 7, 18.0)
record.append(PruningPassIteration(2, 6, 25.0))
record.append(PruningPassIteration(1, 5, 34.0))
record.append(PruningPassIteration(3, 4, 87.0))
mses = [18.0, 25.0, 34.0, 87.0]
sizes = [7, 6, 5, 4]


def test_statistics():
    mses = [record.mse(i) for i in range(len(record))]
    mses_ = [mses[i] for i in range(len(record))]
    gcvs = [record.gcv(i) for i in range(len(record))]
    gcvs_ = [gcv(mses[i], sizes[i], num_samples, penalty)
             for i in range(len(record))]
    rsqs = [record.rsq(i) for i in range(len(record))]
    rsqs_ = [1 - (mses[i] / sst)
             for i in range(len(record))]
    grsqs = [record.grsq(i) for i in range(len(record))]
    grsqs_ = [1 - (record.gcv(i) / gcv(sst, 1, num_samples, penalty))
              for i in range(len(record))]
    assert_list_almost_equal(mses, mses_)
    assert_list_almost_equal(gcvs, gcvs_)
    assert_list_almost_equal(rsqs, rsqs_)
    assert_list_almost_equal(grsqs, grsqs_)
