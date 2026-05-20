# Entry Points

## Selected Submission Notebook

```text
code/inference/gpt-oss-120b-with-tools/aimo-3-gpt-oss-120b-with-tools-version-16.ipynb
```

Run this notebook in the Kaggle AI Mathematical Olympiad - Progress Prize 3 notebook environment with the competition data, GPT-OSS-120B model input, and `aimo-3-utils` utility dataset attached.

The competition entry point is the notebook's `predict()` function:

```python
def predict(id_: pl.DataFrame, question: pl.DataFrame, answer: Optional[pl.DataFrame] = None) -> pl.DataFrame:
```

The notebook registers the callback with Kaggle's inference server:

```python
inference_server = kaggle_evaluation.aimo_3_inference_server.AIMO3InferenceServer(predict)
```

During competition reruns, the notebook executes:

```python
inference_server.serve()
```

For local gateway execution inside Kaggle, the notebook executes:

```python
inference_server.run_local_gateway(
    ("/kaggle/input/ai-mathematical-olympiad-progress-prize-3/test.csv",)
)
```

## Utility Notebook

```text
code/inference/gpt-oss-120b-with-tools/aimo-3-utils.ipynb
```

This utility notebook creates the offline dependency archive used by the selected submission notebook. It downloads Python 3.12 binary wheels and tiktoken encodings, then packages them as:

```text
/kaggle/working/wheels.tar.gz
```

The selected submission notebook expects the archive at:

```text
/kaggle/input/aimo-3-utils/wheels.tar.gz
```

## No Training Entry Point

There is no training entry point. The submitted solution uses pretrained GPT-OSS-120B weights and performs inference only.
