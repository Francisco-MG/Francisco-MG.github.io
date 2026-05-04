# files/README_bertopic.md

## Setup (one-time, conda on Windows)

conda activate bertopic  # or conda activate base
pip install bertopic sentence-transformers pandas matplotlib kaleido

## Run

python ../projet_20/bertopic_ecb_analysis.py
  --corpus files/corpus_ecb.csv \
  --outdir files/results_bertopic/
  
  
python ../files/projet_20/bertopic_ecb_analysis.py --corpus ../files/projet_20/corpus_ecb.csv --outdir ../files/projet_20/results_bertopic/