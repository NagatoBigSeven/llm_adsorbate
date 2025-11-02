#!/bin/zsh
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate llm_adsorbate
python -m src.agent.agent \
  --smiles 'ClC(=O)[O-]' \
  --slab "./notebooks/cu_slab_211.xyz" \
  --user_request "does this fragment stay covalently bound through the carbon to a top site in the middle of the surface?"
