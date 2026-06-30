# Freshness Inspector — Fresh vs Rotten Fruit Classifier

A website where you upload a photo of a fruit and a PyTorch neural network
tells you whether it's **fresh** or **rotten**, plus how confident it is.

This is the Day 14 capstone project: transfer learning (ResNet18) + a real
web app built with Flask, run locally in VS Code.

You do not need any prior experience. Follow the steps in order — don't skip
ahead, each one depends on the last.

---

## What you're building, in plain terms

1. A **training script** (`train_model.py`) that takes thousands of labeled
   fruit photos and teaches a neural network the difference between fresh
   and rotten produce. This step happens once, and takes 10–40 minutes.
2. A **website** (`app.py` + the `templates`/`static` folders) where you, or
   anyone else, can upload a new photo and instantly get a prediction. This
   is what you'll actually use day to day, and it starts in about 2 seconds.

You only have to train once. After that, the website just loads the saved
model and answers instantly.

---

## Step 1 — Install the tools

You need three things on your computer. Install them in this order.

### 1a. Python
Download Python 3.10 or 3.11 from **python.org/downloads**.
On Windows: during install, **check the box "Add Python to PATH"** — this is
the most commonly missed step and causes most beginner errors.

Check it worked by opening a terminal (Command Prompt on Windows, Terminal
on Mac) and typing:
```
python --version
```
You should see something like `Python 3.11.4`. If you see an error, restart
your computer and try again — sometimes the PATH update needs a restart.

### 1b. Visual Studio Code
Download from **code.visualstudio.com**, install it, then open it and
install two extensions (click the squares icon on the left sidebar, search,
click Install):
- **Python** (by Microsoft)
- **Pylance** (by Microsoft)

### 1c. Git (so you can push to GitHub later)
Download from **git-scm.com/downloads**. Defaults are fine during install.

---

## Step 2 — Get the project files into a folder

1. Create a folder somewhere you'll remember, e.g. `Documents/fruit-freshness-app`.
2. Put all the files from this project into it (`app.py`, `train_model.py`,
   `requirements.txt`, the `templates/` folder, the `static/` folder, etc.).
3. Open VS Code, then **File > Open Folder** and select that folder.
4. Open VS Code's built-in terminal: **Terminal > New Terminal**. Every
   command below gets typed into this terminal.

---

## Step 3 — Create a virtual environment

A virtual environment keeps this project's Python packages separate from
everything else on your computer. Always do this for every Python project.

In the VS Code terminal:

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

You'll know it worked because your terminal prompt now starts with `(venv)`.
**Every time you open a new terminal for this project, re-run the activate
line.** VS Code sometimes asks "select this interpreter?" — say yes.

Now install everything the project needs:
```bash
pip install -r requirements.txt
```
This installs PyTorch, torchvision, Flask, and a few helpers. It may take a
few minutes — PyTorch is a large download.

---

## Step 4 — Get the dataset

We'll use the public **"Fruits fresh and rotten for classification"** dataset
on Kaggle, which already has labeled folders of fresh and rotten apples,
bananas, and oranges. (You're welcome to swap in your own fruit photos later
— see Step 7.)

1. Go to **kaggle.com**, make a free account if you don't have one.
2. Search for **"fruits fresh and rotten for classification"** (by sriramr).
3. Click **Download** (you may need to click "Download all" — it's a zip,
   roughly 1.5 GB).
4. Unzip it. Inside you'll find a structure like:
   ```
   dataset/
     train/
       freshapples/
       freshbanana/
       freshoranges/
       rottenapples/
       rottenbanana/
       rottenoranges/
     test/
       freshapples/
       freshbanana/
       freshoranges/
       rottenapples/
       rottenbanana/
       rottenoranges/
   ```
5. Copy the `train` and `test` folders so they sit *inside* the `dataset/`
   folder in your project, replacing the empty placeholder folder that's
   already there. The final path should look like:
   `fruit-freshness-app/dataset/train/freshapples/...`

**Don't have a Kaggle account or don't want one?** Any dataset works as long
as it follows the same pattern: a `train` folder and a `test` folder, each
containing one subfolder per class, and every class folder name starting
with the word `fresh` or `rotten` (e.g. `freshmango`, `rottenmango`). You
could build this yourself by taking 100+ photos of fresh and rotten fruit on
your phone and sorting them into folders.

---

## Step 5 — Train the model

Still in the VS Code terminal, with `(venv)` showing, run:
```bash
python train_model.py
```

What you'll see: progress for each epoch (one full pass through the
training data), with a loss number going down and an accuracy number going
up. That's the network learning. On a laptop CPU this might take 20–40
minutes; if you have an NVIDIA GPU it'll be much faster automatically.

**Want a free GPU instead?** Open the script in Google Colab:
1. Go to **colab.research.google.com**, upload `train_model.py` as a
   notebook cell (or paste its contents into a notebook), and also upload
   your `dataset` folder (zip it first, upload, then unzip in Colab with
   `!unzip dataset.zip`).
2. **Runtime > Change runtime type > T4 GPU**.
3. Run the cell. Once done, download the `model/` folder it produces
   back to your local project folder.

When training finishes you'll have three new files in `model/`:
- `fruit_model.pth` — the trained weights
- `class_names.json` — the list of classes, needed by the website
- `training_curve.png` — a plot of loss/accuracy over time, open it to see
  how training went

If `test` accuracy is below ~85%, the most common fixes are: train for more
epochs (raise `NUM_EPOCHS` in `train_model.py`), or check your dataset
folders aren't mixed up.

---

## Step 6 — Run the website

```bash
python app.py
```

You'll see something like `Running on http://127.0.0.1:5000`. Open that
address in your browser. Drag a fruit photo onto the upload tray (or click
it to browse), and within a second you'll see a verdict — **FRESH** or
**ROTTEN** — the fruit type, and a confidence gauge.

To stop the server, click back in the terminal and press `Ctrl+C`.

**Test it on real photos:** take a photo of a fruit with your phone, email
it to yourself or use a cloud drive to get it onto your computer, then
upload it on the site. This is the real test — does it generalize beyond
the training photos?

---

## Step 7 — Make it your own (optional but recommended)

The whole point of Day 14 is picking a problem *you* care about. A few ways
to extend this:

- **Different fruits or categories.** Swap the dataset for one you've
  collected yourself — e.g. 5 dog breeds, or fresh-vs-spoiled vegetables.
  Just keep the same folder structure (`dataset/train/<class>/`,
  `dataset/test/<class>/`). If your classes aren't "fresh/rotten" pairs,
  open `app.py` and simplify the `parse_class_name` function to just return
  the class name directly.
- **More classes.** Add more fruit subfolders, retrain, done — the code
  automatically adapts to however many class folders it finds.
- **Better accuracy.** In `train_model.py`, try unfreezing `model.layer3`
  as well as `layer4`, or increase `NUM_EPOCHS`.

---

## Step 8 — Push it to GitHub

1. Create a free account at **github.com** if you don't have one.
2. Click **New repository**, name it e.g. `fruit-freshness-app`, leave it
   public or private, don't initialize with a README (you already have one).
3. Back in the VS Code terminal:
   ```bash
   git init
   git add .
   git commit -m "Fresh vs rotten fruit classifier"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/fruit-freshness-app.git
   git push -u origin main
   ```
4. Refresh your GitHub repo page — your code is now live.

Note: the `.gitignore` file already excludes the dataset and trained model
weights from being uploaded (they're too large for a normal repo). Anyone
who clones your repo will need to redo Steps 4–5 to get a working model.
If you want others to be able to use the app without retraining, see the
note on Git LFS or releasing the `.pth` file as a GitHub Release asset.

---

## Project structure reference

```
fruit-freshness-app/
├── app.py                  # Flask website — run this day-to-day
├── train_model.py          # Trains the model — run this once (or whenever you change the dataset)
├── requirements.txt        # Python packages this project needs
├── dataset/                # Your training photos go here (not included)
│   ├── train/
│   └── test/
├── model/                  # Created automatically by train_model.py
│   ├── fruit_model.pth
│   ├── class_names.json
│   └── training_curve.png
├── templates/
│   └── index.html          # The page structure
└── static/
    ├── css/style.css       # Look and feel
    ├── js/script.js        # Upload + result logic
    └── uploads/            # Photos you upload through the site land here temporarily
```

## Troubleshooting

| Problem | Likely fix |
|---|---|
| `python: command not found` | Python isn't installed or wasn't added to PATH. Reinstall and check the PATH box. |
| `ModuleNotFoundError: No module named 'torch'` | Your venv isn't activated, or `pip install -r requirements.txt` didn't finish. Re-run both. |
| Website says "No trained model found" | You haven't run `train_model.py` yet, or it failed partway. Check the `model/` folder for `fruit_model.pth`. |
| Training accuracy stuck very low | Double check your dataset folders — each class folder should contain only that class's images, and folder names should start with `fresh` or `rotten`. |
| `CUDA out of memory` | Lower `BATCH_SIZE` in `train_model.py`, e.g. to 16. |
| Predictions look random/wrong on your own photos | Take photos with the fruit centered and well-lit, similar in style to training photos. Very different angles/backgrounds confuse any model trained on a few thousand images. |
