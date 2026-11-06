# Step-by-step: publish ClaimSense-LLM on GitHub

## Option A — Website only (no coding tools)
1. Sign in at https://github.com → click **+** (top right) → **New repository**.
2. Repository name: `claimsense-llm`
   Description: `Domain-adapted LLM for insurance claims triage, coverage reasoning & fraud-signal detection (QLoRA, Mistral-7B).`
3. Select **Public**. Do NOT tick "Add a README", .gitignore or license (they are in the ZIP). Click **Create repository**.
4. On the empty repo page click **uploading an existing file**.
5. Unzip `claimsense-llm.zip` on your computer, open the `claimsense-llm` folder, select ALL its contents (including folders) and drag them into the browser.
   - `.gitignore` is hidden on Mac/Windows: show hidden files (Mac: Cmd+Shift+. ; Windows: View → Hidden items), or create it afterwards with **Add file → Create new file** named `.gitignore` and paste its text.
6. Commit message: `Initial commit: ClaimSense-LLM`. Click **Commit changes**.

## Option B — Command line
```bash
cd claimsense-llm
git init
git add .
git commit -m "Initial commit: ClaimSense-LLM"
git branch -M main
git remote add origin https://github.com/<your-username>/claimsense-llm.git
git push -u origin main
```

## After uploading (makes it look professional)
1. Repo page → gear icon next to **About** → add the description, and Topics:
   `llm`, `insurance`, `insurtech`, `nlp`, `fine-tuning`, `qlora`, `mistral`, `fraud-detection`, `claims`, `generative-ai`.
2. Edit `README.md` §11 with your LinkedIn and email; edit `LICENSE` with your full name.
3. Pin it: your profile → **Customize your pins** → select `claimsense-llm`.
4. (Optional) Run training on Google Colab (Runtime → GPU), then replace the README §7 numbers with your real results.
5. (Optional) Upload the trained adapter to Hugging Face Hub and link it in the README.
