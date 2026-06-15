import pyvsag
import numpy as np
import json
from ..base.module import BaseANN


class Vsag(BaseANN):
    index_name = None
    build_param_key = None
    search_param_key = None

    def __init__(self, metric, dim, method_param):
        self._metric = {"euclidean": "l2", "angular": "l2"}[metric]
        self._normalize = metric == "angular"
        self._dim = dim
        self._params = method_param
        self._ef = 10
        self.name = "vsag-%s (%s)" % (self.index_name, self._params)

    def _index_params(self, X):
        dim = X.shape[1]
        if self._dim != dim:
            raise ValueError("Configured dimension %d does not match data dimension %d" % (self._dim, dim))

        return {
            "dtype": "float32",
            "metric_type": self._metric,
            "dim": dim,
            self.build_param_key: self._build_params(len(X)),
        }

    def _build_params(self, num_elements):
        raise NotImplementedError

    def _prepare_vectors(self, X):
        X = np.asarray(X, dtype=np.float32)
        if not self._normalize:
            return np.ascontiguousarray(X)

        X = np.array(X, dtype=np.float32, copy=True, order="C")
        norms = np.linalg.norm(X, axis=1)
        zero_norms = norms == 0
        if np.any(zero_norms):
            X[zero_norms] = 1.0 / np.sqrt(X.shape[1])
            norms[zero_norms] = 1.0
        X /= norms[:, np.newaxis]
        return X

    def _prepare_query(self, v):
        v = np.asarray(v, dtype=np.float32)
        if not self._normalize:
            return np.ascontiguousarray(v)

        norm = np.linalg.norm(v)
        if norm == 0:
            return np.full(v.shape, 1.0 / np.sqrt(v.shape[0]), dtype=np.float32)
        return np.ascontiguousarray(v / norm, dtype=np.float32)

    def fit(self, X):
        X = self._prepare_vectors(X)
        index_params = self._index_params(X)
        print(index_params)
        self._index = pyvsag.Index(self.index_name, json.dumps(index_params))
        self._index.build(
            vectors=X,
            ids=np.arange(len(X), dtype=np.int64),
            num_elements=len(X),
            dim=X.shape[1],
        )

    def set_query_arguments(self, ef):
        self._ef = ef
        self.name = "vsag-%s (%s, efSearch=%s)" % (self.index_name, self._params, self._ef)

    def query(self, v, n):
        search_params = {self.search_param_key: {"ef_search": self._ef}}
        ids, dists = self._index.knn_search(
            vector=self._prepare_query(v),
            k=n,
            parameters=json.dumps(search_params),
        )
        return ids


class VsagHNSW(Vsag):
    index_name = "hnsw"
    build_param_key = "hnsw"
    search_param_key = "hnsw"

    def _build_params(self, num_elements):
        params = {
            "max_degree": self._params["M"],
            "ef_construction": self._params["ef_construction"],
            "ef_search": self._params["ef_construction"],
            "max_elements": num_elements,
            "use_static": False,
        }

        if self._params.get("use_int8", 0) != 0:
            params["sq_num_bits"] = self._params["use_int8"]
        if "rs" in self._params:
            params["redundant_rate"] = self._params["rs"]
        return params


class VsagHGraph(Vsag):
    index_name = "hgraph"
    build_param_key = "index_param"
    search_param_key = "hgraph"

    def _build_params(self, num_elements):
        params = {
            "max_degree": self._params["max_degree"],
            "ef_construction": self._params["ef_construction"],
        }
        if "base_quantization_type" in self._params:
            params["base_quantization_type"] = self._params["base_quantization_type"]
        if "alpha" in self._params:
            params["alpha"] = self._params["alpha"]
        return params
