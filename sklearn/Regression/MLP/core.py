from debugpy import connect
import numpy as np
import pandas as pd
import sys

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import make_column_selector
from joblib import dump



class RuidingMLEngine:
    def __init__(self):
        self.client = None
        self.data = None
        self.pipeline = None
        self.splitter = None
        self.output_generator = None
        self.info = {}

        self._fitted = False
        self.data_train_ = None
        self.data_test_ = None

    def run(self):        

        self.client = connect()

        self.data_train_, self.data_test_ = self.splitter(self.data)

        target_col = self.info.get("target_col")
        weight_col = self.info.get("weight_col")
        feature_cols = [var for var in self.data_train_.columns if var not in [target_col, weight_col]]

        X_train = self.data_train_[feature_cols]
        y_train = self.data_train_[target_col] if target_col is not None else None
        weight_train = self.data_train_[weight_col] if weight_col is not None else None
        self.pipeline.fit(X_train, y_train, **{f"{self.info.get('model_name')}__sample_weight": weight_train})
        self._fitted = True

        save_path = self.info.get("save_path")
        if save_path is not None:
            dump(self.pipeline, save_path)

        
    def output_metrics(self):
        if not self._fitted:
            raise ValueError("Model not fitted yet")
        
        res = self.output_generator.output()
        return res

class RuidingMLEngineBuilder:
    
    def __init__(self):
        self.engine = RuidingMLEngine()
    def set_client(self, client):
        self._engine.client = client
    def set_data(self, data) :
        self._engine.data = data
    def set_splitter(self, splitter):
        self._engine.splitter = splitter
    def set_pipeline(self, pipeline):
        self._engine.pipeline = pipeline
    def set_pipeline_componnent(self, name, component):
        pipeline_element = (name, component)
        if self._engine.pipeline is None:
            raise ValueError("Pipeline not set yet")
        elif len(self.engine.pipeline) == 0 and hasattr(component, "fit"):
            self._engine.pipeline.steps.append(pipeline_element)
        elif len(self.engine.pipeline) > 0 and hasattr(component, "fit") \
                and hasattr(self.engine.pipeline[-1][1], "transform"):
            self._engine.pipeline.steps.append(pipeline_element)
        else:
            raise ValueError("Pipeline not set yet")
    def set_info(self, k, v=None):
        if isinstance(k, dict):
            self._engine.info.update(k)
        elif isinstance(k, str):
            self._engine.info[k] = v
        else:
            raise ValueError("Invalid input")   
    def set_output_generator(self, output_generator):
        output_generator.set_engine(self._engine)
        self._engine.output_generator = output_generator
    @property
    def engine(self):
        return self._engine
    
class RuidingMLEngineDirector:
    def __init__(self, data_info, model_info, cluster_info):
        self._data_info = data_info
        self._model_info = model_info
        self._cluster_info = cluster_info
        self._builder = RuidingMLEngineBuilder()
    def build(self):
        cluster_backend = self._cluster_info.get("backend")
        address = self._cluster_info.get("address")
        config = self._cluster_info.get("config")
        
        client = get_client(cluster_backend, address, config)
        self._builder.set_client(client)
        self._builder.set_info("cluster_backend", cluster_backend)

        path = self._data_info.get("path")
        data_backend = self._data_info.get("backend")
        columns = self._data_info.get("columns")

        if columns is not None:
            data = read_df(backend=data_backend, path=path, columns=columns)
        else:
            data = read_df(backend=data_backend, path=path)

        self._builder.set_data(data)
        target_col = self._data_info.get("target_col")
        weight_col = self._data_info.get("weight_col")
        setID_col = self._data_info.get("setID_col")
        enums = self._data_info.get("enums")
        valid_value = self._data_info.get("valid_value")
        self._builder.set_info("target_col", target_col)
        self._builder.set_info("weight_col", weight_col)
        splitter = get_splitter(data_backend, 
                                target_col=target_col,
                                weight_col=weight_col,
                                setID_col=setID_col,
                                enums=enums,
                                valid_value=valid_value)
        self._builder.set_splitter(splitter)

        pipeline = Pipeline(steps=[])
        self._builder.set_pipeline(pipeline)
        model_name = self._model_info.get("name")
        self._builder.set_info("model_name", model_name)

        lower = model_name
        if ("xgb" in lower) or ("tree" in lower) or ("forest" in lower):
            encoder = LabelEncoder()
        else:
            encoder = OneHotEncoder()
        ct = ColumnTransformer([('identical', encoder, make_column_selector(dtype_include=object))],remainder='passthrough')
        self._builder.set_pipeline_componnent("preprocessor", ct)

        task = self._model_info.get("task")
        model_backend = self._model_info.get("backend")
        model_args = self._model_info.get("args")
        origin_model = get_model(model_name, model_backend, model_args)

        model = origin_model
        self._builder.set_pipeline_componnent(model_name, model)
        self._builder.set_info("save_path", self._model_info.get("save_path"))
        genarator = get_output_generator(model_backend, task)
        self._builder.set_output_generator(genarator)

        return self._builder.engine
    
def get_splitter(backend, *args, **kwargs):
    if backend == "pandas":
        from sklearn.model_selection import train_test_split
        splitter = train_test_split
    elif backend == "dask":
        from dask_ml.model_selection import train_test_split
        splitter = train_test_split
    elif backend == "spark":
        from pyspark.sql import DataFrame
        splitter = DataFrame.randomSplit
    else:
        raise ValueError("Backend not supported")
    return splitter
    
def get_client(backend, address, config):
    if backend == "dask":
        from dask.distributed import Client
        client = Client(address, **config)
    elif backend == "spark":
        from pyspark.sql import SparkSession
        spark = SparkSession.builder.appName("ruiding_ml").getOrCreate()
        client = spark
    else:
        raise ValueError("Backend not supported")
    return client

def get_output_generator(backend, task):
    if backend == "pandas":
        module = "pandas"
    elif backend == "dask":
        module = "dask"
    elif backend == "spark":
        module = "pyspark"
    else:
        raise ValueError("Backend not supported")
    
    if task == "regression":
        generator = getattr(sys.modules[module], "DataFrame")
    elif task == "classification":
        generator = getattr(sys.modules[module], "DataFrame")
    else:
        raise ValueError("Task not supported")
    return generator

def read_df(path, backend, *args, **kwargs):
    if backend == "pandas":
        module = "pandas"
    elif backend == "dask":
        module = "dask.dataframe"
    elif backend == "spark":
        module = "pyspark.pandas"
    else:
        raise ValueError("Backend not supported")
    
    extension_name = path.split(".")[-1]
    reader = getattr(sys.modules[module], "read_" + extension_name)
    data = reader(path, *args, **kwargs)
    return data

def get_model(name, backend, args):
    module_name = backend + "_mdls"
    import importlib
    module = importlib.import_module(f"ruiding_ml.{module_name}")
    try:
        modelClass = getattr(module, name)
    except KeyError:
        raise ValueError("Model not supported")
    except AttributeError:
        raise ValueError("Model not supported")
    return modelClass(**args)


def ruiding_ml_training(data_info, model_info, cluster_info):
    director = RuidingMLEngineDirector(data_info, model_info, cluster_info)
    ml_engine = director.build()
    ml_engine.run()
    return ml_engine.output_metrics()