Dog Breed Models (299×299, IRv2 + Xception)
Pretrained Keras models for the Dog Breed Identification task at 299×299, plus exact class order and an ensemble inference script. Large artifacts are hosted on Kaggle; this repo contains code, config, and instructions to reproduce submissions.

Weights and data
Kaggle dataset: arpitshivhare2016/dogbreed-models-299. Download with the CLI or attach it in Kaggle Notebooks.

Download locally:

bash
pip install kaggle
kaggle datasets download -d arpitshivhare2016/dogbreed-models-299
unzip dogbreed-models-299.zip -d artifacts
This mirrors Kaggle’s standard dataset download flow and places files under artifacts/.

Contents
config.json — Model config and preprocessing spec (IRv2 and Xception at 299×299).

class_index.json — Exact training order of 120 breeds to map logits → labels.

ensemble_infer.py — Loads both models, runs weighted soft-vote, writes submission.csv.

README.md — Usage instructions and examples.

Weights on Kaggle — best_irv2.keras and best_iv4.keras hosted in the dataset above.

Quick start (local)
Install dependencies:

bash
pip install tensorflow pandas numpy
Place artifacts:

After unzip, ensure artifacts/ contains: best_irv2.keras, best_iv4.keras, class_index.json, config.json.

Run ensemble:

bash
python ensemble_infer.py \
  --test_root /path/to/test/images \
  --sample_csv /path/to/sample_submission.csv \
  --classes artifacts/class_index.json \
  --model_irv2 artifacts/best_irv2.keras \
  --model_iv4  artifacts/best_iv4.keras \
  --img_size 299 --batch_size 16 --w_irv2 0.6 --w_iv4 0.4 \
  --output submission.csv
This follows the competition format: id plus 120 breed columns with probabilities.

Quick start (Kaggle Notebook)
Create a Notebook and “Add data” → add arpitshivhare2016/dogbreed-models-299. Files appear under /kaggle/input/dogbreed-models-299/.

Example paths:

python
CLASSES = "/kaggle/input/dogbreed-models-299/class_index.json"
IRV2    = "/kaggle/input/dogbreed-models-299/best_irv2.keras"
IV4     = "/kaggle/input/dogbreed-models-299/best_iv4.keras"
Use the competition’s provided test directory and sample_submission.csv as usual.

Preprocessing details
InceptionResNetV2: tensorflow.keras.applications.inception_resnet_v2.preprocess_input.

Xception: tensorflow.keras.applications.xception.preprocess_input.

Images resized to 299×299 RGB; probabilities numerically clipped and renormalized before CSV.

Tips
Title/id rules for Kaggle metadata: titles 6–50 chars; id as username/slug with lowercase and dashes. Useful if publishing personal variants.

Use Kaggle for large artifacts and GitHub for code to avoid Git LFS and GitHub file-size limits. Link the Kaggle dataset in this README as shown above.

References
Kaggle CLI docs for dataset download and versioning.

Kaggle Datasets overview and Notebook data mounting behavior.

Dog Breed Identification competition format and submission requirements.

how to deletea file from repo

how to deletea file from repo
Yes. Use either the command line or the GitHub web UI depending on whether the local file should also be deleted.

Delete from repo and disk (CLI)
Removes the file locally and from the repository history moving forward:

bash
git rm path/to/file.ext
git commit -m "Delete file.ext"
git push
This stages a deletion, commits it, and updates the remote; by default git rm deletes from both the index and working tree per Git docs.

Remove from repo but keep locally (CLI)
Stop tracking the file while keeping the local copy:

bash
git rm --cached path/to/file.ext
echo path/to/file.ext>>.gitignore
git add .gitignore
git commit -m "Stop tracking file.ext; keep locally"
git push
The --cached flag removes only from the index; adding to .gitignore prevents re-adding it later, which is the standard pattern to keep local files but remove them from the repo.

Delete a folder
From repo and disk:

bash
git rm -r path/to/folder
git commit -m "Delete folder"
git push
From repo only (keep locally):

bash
git rm -r --cached path/to/folder
echo path/to/folder/>>.gitignore
git add .gitignore
git commit -m "Stop tracking folder"
git push
Recursive options are supported for directories as documented in git-rm references.

Delete via GitHub web UI
Open the file in the repository, click the three-dots menu or Edit, then choose Delete this file, add a commit message, and commit directly on the branch; this is the documented UI path for single-file deletes on GitHub.

If the trash icon isn’t visible for text files, use Edit → Delete this file, or perform the deletion locally with the CLI and push, as per GitHub docs and community notes.

Note:

For sensitive data already committed, use git filter-repo or GitHub’s removal guidance to rewrite history if necessary; simple git rm affects only future commits and won’t purge old versions from history.

C:\Users\KIIT\Desktop\kaggle_upload>git add .gitignore
fatal: pathspec '.gitignore' did not match any files

C:\Users\KIIT\Desktop\kaggle_upload>

C:\Users\KIIT\Desktop\kaggle_upload>git add .gitignore fatal: pathspec '.gitignore' did not match any files C:\Users\KIIT\Desktop\kaggle_upload>
The error means there’s no .gitignore file in the current directory yet; create it first, then add and commit it.

Create .gitignore on Windows CMD
Create an empty file:

type NUL > .gitignore

Or open Notepad to create/edit:

notepad .gitignore (save when prompted)

Or create with initial patterns:

echo kaggle_upload*.keras>>.gitignore

echo *.zip>>.gitignore

echo pycache>>.gitignore

Add and commit it
git add .gitignore

git commit -m "Add .gitignore to exclude large/generated files"

git push

If files are already tracked
To stop tracking files already committed, remove from the index and keep locally:

git rm --cached path\to\file.ext

git rm -r --cached path\to\dir

git commit -m "Stop tracking large files"

git push

Notes:

After adding ignore patterns, Git won’t retroactively untrack existing files; use git rm --cached for those, as documented in gitignore/git-rm references.

If CMD has trouble creating dotfiles, using notepad .gitignore or type NUL > .gitignore reliably creates it on Windows per community guidance.

Related
Create a .gitignore file in this repo from Command Prompt
Show tracked files to confirm if .gitignore is already tracked
Add node_modules to .gitignore and stop tracking it
Use git rm --cached to remove an already tracked file
Set up a global .gitignore for Windows and Git Bash






