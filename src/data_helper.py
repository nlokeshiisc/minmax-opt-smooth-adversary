import torch
from src.abstract import abs_data
import constants as constants
import itertools

class SyntheticData(abs_data.Data):
    def __init__(self, X, y, Z, Beta, Z_ids, *args, **kwargs) -> None:
        super(SyntheticData, self).__init__(X, y, Z, Beta, Z_ids, *args, **kwargs)
    
    def apply_recourse(self, data_ids, betas:torch.Tensor):
        """Applies recourse to the specified data is and returns the recoursed x
        Args:
            data_ids ([type]): [description]
            betas ([type]): [description]
        Returns:
            [type]: [description]
        """
        _, _, z, _ = self.get_instances(data_ids)
        assert z.shape() == betas.shape(), "Why the hell are the shapes inconsistent?"
        return torch.multiply(z, betas)

    @property
    def _list_beta_dims(self):
        """Returns a list of numbers to be used by g_phi to predict the betas.
        """
        return [2] * self._Beta.shape[1]
    
    @property
    def _num_reg_fns(self) -> int:
        return 1
    
    @property
    def _list_regfn_dims(self):
        return [2]

    def _get_unq_beta(self) -> torch.Tensor:
        return torch.unique(self.Beta, dim=0)
        # if self._num_beta == 6:
        #     return torch.LongTensor(list(set(itertools.permutations([1,1,1,0,0,0]))))
        # elif self._num_beta == 4:
        #     return torch.LongTensor(list(set(itertools.permutations([1,1,0,0]))))

    def register_x(self, x) -> torch.Tensor:
        mid = int(len(x)/2)
        if torch.any(x[0:mid] > constants.TOL):
            return torch.LongTensor([0])
        else:
            return torch.LongTensor([1])

    def respond_queries(self, queries):
        X = []
        y = []
        zids = []
        Beta = []
        for query in queries:
            x = self.get_x_for_zbeta(self._Z[query[0]], query[1])
            X.append(x)
            y.append(self._y[self.get_siblings(query[0])[0][0]])
            zids.append(query[0])
            Beta.append(query[1])

        X = torch.stack(X, dim=0)
        y = torch.stack(y, dim=0).squeeze()
        zids = torch.Tensor(zids).to(dtype=int)
        Beta = torch.stack(Beta, dim=0)

        new_ids = self._add_new_data(X=X, y=y, Beta=Beta, Zids=zids)
        return new_ids
    
    def get_x_for_zbeta(self, z, beta, **kwargs):
        return torch.mul(z, beta)  
        
class SyntheticDataHelper(abs_data.DataHelper):
    def __init__(self, train, test, val, train_test_data=None, ds_name=None) -> None:
        super(SyntheticDataHelper, self).__init__(train, test, val,  train_test_data, ds_name)

