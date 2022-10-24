import numpy as np
import bentoml
from bentoml.io import NumpyNdarray

zoomcamp_homework_runner = bentoml.sklearn.get("mlzoomcamp_homework:jsi67fslz6txydu5").to_runner()

svc = bentoml.Service("zoomcamp_homework", runners=[zoomcamp_homework_runner])

@svc.api(input=NumpyNdarray(), output=NumpyNdarray())
def classify(input_series: np.ndarray) -> np.ndarray:
    result = zoomcamp_homework_runner.predict.run(input_series)
    return result