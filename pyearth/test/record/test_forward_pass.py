from pyearth._record import ForwardPassRecord, ForwardPassIteration
from pyearth._util import gcv
from ..testing_utils import assert_list_almost_equal


num_samples = 1000
num_variables = 10
penalty = 3.0
sst = 100.0
varnames = ['x' + str(i) for i in range(num_variables)]
record = ForwardPassRecord(num_samples, num_variables,
                           penalty, sst, varnames)
record.append(ForwardPassIteration(0, 3, 3, 63.0, 3))
record.append(ForwardPassIteration(0, 3, 14, 34.0, 5))
record.append(ForwardPassIteration(3, 6, 12, 18.0, 7))
mses = [sst, 63.0, 34.0, 18.0]
sizes = [1, 3, 5, 7]


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
