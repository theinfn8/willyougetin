# Will You Get In?

A fine-tuned DistilBERT model that predicts whether a graduate school
application was accepted or rejected, from the kind of details applicants post
on [The GradCafe](https://www.thegradcafe.com/). It's measured against a
two-layer neural network written from scratch in NumPy, and served as an int8
ONNX model from a small Flask app.

- **Live demo:** <https://chaostek.dev/willyougetin/>
- **Project write-up:** <https://chaostek.dev/data/data1.html>

> **This is a portfolio project, not admissions advice.** The data is
> self-reported and says nothing about essays, letters or research fit, and the
> model leans heavily on degree type: about 31% of PhD results in the test data
> were acceptances, but it predicts only 19% of them as accepted (Master's: 76%
> actual, 85% predicted).

## Results

Held-out test sets (the 20% of results each model never saw in training).
Accepted is the positive class.

| Model | Test rows | Accuracy | Precision | Recall | F1 | Size |
|---|---|---|---|---|---|---|
| Always predict "Rejected" | 7,853 | 55.2% | – | – | – | – |
| Two-layer NumPy network (baseline) | 7,778 | 71.8% | 76.0% | 54.6% | 63.5% | 49 weights |
| DistilBERT, PyTorch fp32 | 7,853 | 79.1% | 80.0% | 71.2% | 75.3% | 268 MB |
| DistilBERT, ONNX int8 (served) | 7,853 | 79.1% | 79.9% | 71.1% | 75.3% | 67 MB |

The baseline filters the data slightly differently, so its test set is a little
smaller; the two are close but not identical. The int8 model agrees with the
fp32 model on 97.7% of test rows. Full numbers are in
[`artifacts/eval_results.json`](artifacts/eval_results.json).

## How it works

**Baseline (`baseline/`).** A 6 → 6 → 1 network with sigmoid activations, MSE
loss, full-batch gradient descent and early stopping, with the forward pass and
backpropagation written by hand. Inputs are GPA, three GRE scores, degree type
and citizenship. Missing values are filled with training-set medians and then
standardized.

**DistilBERT (`model/`).** Each applicant is written into one text template:

```
Program: Computer Science
University: Johns Hopkins University
Comments: Two years of research experience, one publication.
Degree: PhD
Citizenship: International
GPA: 3.87
GRE Quant: Not reported
GRE Verbal: Not reported
GRE AW: Not reported
```

Missing scores are written as "Not reported" rather than imputed, because on
GradCafe not reporting a score can itself carry signal. Outcome words in
comments ("accepted", "rejected", "waitlisted"…) are replaced with `[redacted]`
so the model can't read the answer. `distilbert-base-uncased` is then
fine-tuned for 3 epochs (AdamW, learning rate 2e-5, batch size 16, max 256
tokens) on a stratified 80/20 split.

**Experiment tracking.** Every run is logged to one MLflow experiment,
`will-you-get-in`: parameters, per-epoch curves, test metrics and the model
files.

**Serving (`service/`).** The model is exported to ONNX and dynamically
quantized to int8. The Flask app needs only `onnxruntime`, `tokenizers` and
`numpy`, with no PyTorch, and runs in about 175 MB of memory. Form input is
checked in the browser and again on the server (realistic score ranges, fixed
choices, length limits). The page also shows the exact text the model read.
`tests/test_parity.py` checks that the service builds that text exactly as
training did, for every row of the dataset.

## Repository layout

```
baseline/        NumPy two-layer network + converter for its input file
model/           DistilBERT fine-tuning (train_model.py) and PyTorch inference
scripts/         evaluate.py, log_existing_model.py, package_model.sh
service/         Flask app, templates, static files, serving requirements
tests/           validation, routes, model, and training/serving parity tests
artifacts/       evaluation results and training history (ONNX files ignored)
tracking.py      shared MLflow setup
Dockerfile, compose.yaml   container for the service
```

## Data

The datasets are **not included**. Both models use `data/cleaned_gradcafe.json`:
GradCafe results scraped and cleaned in earlier coursework. The scraper is part
of [theinfn8/deployanywhere](https://github.com/theinfn8/deployanywhere). The
file is a JSON array of records; these fields are used:

| Field | Example | Notes |
|---|---|---|
| `outcome` | `"Accepted"` | rows other than Accepted/Rejected are dropped |
| `url` | GradCafe result URL | used to remove duplicates |
| `llm-generated-program` | `"Computer Science"` | normalized program name |
| `llm-generated-university` | `"Johns Hopkins University"` | normalized university name |
| `comments` | free text | may be empty |
| `Degree` | `"PhD"`, `"Master's"`, `"PsyD"` | |
| `US/International` | `"American"`, `"International"`, `"Other"` | |
| `GPA`, `GRE`, `GRE V`, `GRE AW` | numbers or null | |

The baseline reads a smaller JSON Lines file made from the same data:

```bash
python baseline/convert_input.py --input data/cleaned_gradcafe.json \
    --output baseline/gradcafe_module12_input.jsonl
```

## Running it

Set up a Python 3.12 environment for training and evaluation:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt     # see the file for CPU-only / AMD GPU PyTorch
```

**Baseline:** about a minute on a CPU. It logs to MLflow and writes `mse_curve.png`.

```bash
cd baseline && python neural_network.py && cd ..
```

**Fine-tune DistilBERT:** about 30 minutes on a mid-range GPU, and much longer
on a CPU. It saves the model to `model/saved_model/` and logs the run. For a
quick check that everything works, add `--max-rows 200 --epochs 1 --save-dir /tmp/smoke`.

```bash
python model/train_model.py
```

**Evaluate and export:** rebuilds the exact test split, scores the PyTorch
model, exports `artifacts/model.onnx` and `artifacts/model.int8.onnx`, and
writes `artifacts/eval_results.json`.

```bash
python scripts/evaluate.py
python scripts/log_existing_model.py   # optional: log the saved model to MLflow without retraining
```

**View the MLflow runs:**

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

**Tests.** The parity and model tests skip automatically when the dataset or
the packaged model isn't present.

```bash
pytest
```

## Running the web service

Package the three files the service needs (`model.int8.onnx`,
`tokenizer.json`, `metadata.json`) into `service/model_files/`, then start the
container:

```bash
scripts/package_model.sh
docker compose up -d --build
curl http://127.0.0.1:8090/health      # {"model_loaded":true,"status":"ok"}
```

The container listens on `127.0.0.1` only (set `WYGI_PORT` to change the port),
is capped at 400 MB of memory, and runs with a read-only filesystem. The model
directory is mounted rather than built into the image.

To serve it under a path such as `/willyougetin/` behind Apache, proxy the path
and send the prefix, so the app's links keep it:

```apache
RedirectMatch 301 ^/willyougetin$ /willyougetin/
<Location /willyougetin/>
    ProxyPass        http://127.0.0.1:8090/
    ProxyPassReverse http://127.0.0.1:8090/
    RequestHeader set X-Forwarded-Prefix "/willyougetin"
    RequestHeader set X-Forwarded-Proto  "https"
</Location>
```

## Background

This project brings together three assignments from Johns Hopkins EN.605.256
(Modern Software Concepts): experiment tracking with MLflow
(Module 11), a neural network built from scratch (Module 12), and fine-tuning
and deploying a language model (Module 13).

Chris Carson · [chaostek.dev](https://chaostek.dev/)
