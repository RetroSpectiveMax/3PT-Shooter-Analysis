"""Build the follow-up notebooks (03 to 05) without touching notebooks 01 and 02.

Usage: python src/build_followups.py
Then execute with: jupyter nbconvert --to notebook --execute --inplace notebooks/<name>.ipynb
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_notebooks as b  # noqa: E402
from cold_cells import COLD  # noqa: E402
from q1b_cells import Q1B  # noqa: E402
from roles_cells import ROLES  # noqa: E402

if __name__ == "__main__":
    b.build("03_q1b_margin_control.ipynb", Q1B)
    b.build("04_cold_streak.ipynb", COLD)
    b.build("05_q3_roles_lineups.ipynb", ROLES)
    print("built 03, 04, 05")
