---
title: Cricket Pressure Performance Intelligence Dashboard
emoji: 🏏
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# INTERACTIVE CRICKET PRESSURE PERFORMANCE DASHBOARD

## 1. Overview
The **Interactive Cricket Pressure Performance Dashboard** is an advanced sports analytics and player scouting application designed to evaluate player reliability under high and extreme pressure across **352,625 legal deliveries** and **1,540 matches** (2022–2026).

It integrates empirical performance metrics with a GPU-trained **XGBoost & CatBoost Machine Learning model** (NVIDIA GeForce RTX 4050, CUDA 13.1) to generate delivery-level Expected Runs ($xRuns$), Boundary Probabilities, and Wicket Hazards.

---

## 2. Core Architecture & Player Universe

### A. All-Player Universe (1,635 Players)
- Every player with match records in any of the 7 supported competitions (**IPL, T20I, Big Bash League, SA20, Major League Cricket, International League T20, Lanka Premier League**) is searchable and profileable.
- Over 999 batters and 740 bowlers have $\ge 15$ pressure deliveries in the multi-league corpus.

### B. Dedicated Challenge Shortlist (The Locked 19 Players)
- The fixed 19-player candidate pool is independently accessible via a dedicated quick filter toggle and the Leaderboards tab.
- Their performance ratings, ranks, and scouting justifications match **100% identically** with the research deliverables (`01_Top_10_Batsmen.xlsx`, `02_Top_10_Bowlers.xlsx`, and `04_Methodology_and_Research_Report.pdf`).
- Pure batters/wicket-keepers (e.g. Virat Kohli, Dinesh Karthik, Rohit Sharma, MS Dhoni) are role-gated with explicit **`Non-Bowler (Excluded)`** badges in the bowling section, preventing distorted rankings.

---

## 3. Football-Style Radar Pentagon Methodology

Every player is represented by a dynamic 5-axis polygon normalized strictly on a **0 to 100 scale** using competition-calibrated reference distributions.

### Batter Radar Dimensions (5 Axes)
1. **Batting Efficiency (0–100):**
   - **Formula:** $0.40 \times \text{clip}(50 + (P_{SR} - 130) \times 0.55, 0, 100) + 0.30 \times \text{clip}(50 + (P_{Runs}/P_{Balls} - 1.30) \times 55, 0, 100) + 0.30 \times \text{clip}(50 + RAE_{ball} \times 80, 0, 100)$
   - Combines Pressure Strike Rate, scoring rate per ball, and ML Runs Above Expected ($RAE$).
2. **Dot-Ball Control (0–100):**
   - **Formula:** $\text{clip}(100 - (P_{Dot\%} - 20.0) \times 2.0, 0, 100)$
   - Inverted transformation: lower dot-ball percentage yields a higher score (elite death batters keep dots under 25%).
3. **Pressure Contribution (0–100):**
   - **Formula:** $0.45 \times \min(100, \frac{P_{Runs}}{400} \times 100) + 0.35 \times \text{clip}(50 + (P_{SR} - N_{SR}) \times 0.75, 0, 100) + 0.20 \times \text{Extreme Share}$
   - Measures run volume in high pressure, strike rate uplift over normal baseline ($\Delta SR$), and extreme pressure output.
4. **Consistency (0–100):**
   - **Formula:** Scaled coefficient of variation ($CV = \sigma_{SR} / \mu_{SR}$) of innings strike rates across pressure appearances, adjusted for sample size.
5. **Chase & Match Impact (0–100):**
   - **Formula:** $0.60 \times \text{clip}(50 + (Chase_{SR} - 125) \times 0.55, 0, 100) + 0.40 \times \min(100, \frac{Chase_{Runs}}{250} \times 100)$
   - Measures performance in Innings 2 run chases and steep required run rate contexts.

### Bowler Radar Dimensions (5 Axes)
1. **Economy Control (0–100):**
   - **Formula:** $\text{clip}(100 - (P_{Eco} - 6.0) \times 12.5, 0, 100)$
   - Inverted transformation relative to T20 death overs average (9.60 RPO). Lower economy produces a higher score.
2. **Dot-Ball Ability (0–100):**
   - **Formula:** $\text{clip}((P_{Dot\%} - 15.0) \times 2.85, 0, 100)$
   - Higher dot-ball suppression under pressure receives higher ratings.
3. **Wicket Threat (0–100):**
   - **Formula:** $0.60 \times \text{clip}(100 - (P_{Balls/Wkt} - 8.0) \times 3.5, 0, 100) + 0.40 \times \text{clip}(50 + WAE \times 10, 0, 100)$
   - Combines pressure strike rate with Wickets Above Expected ($WAE$).
4. **Consistency (0–100):**
   - **Formula:** Spell-to-spell stability measured by economy rate variance across pressure spells.
5. **Pressure Impact (0–100):**
   - **Formula:** $0.60 \times \text{clip}(50 + \frac{RunsSaved}{P_{Balls}} \times 80, 0, 100) + 0.40 \times \text{Extreme Suppression}$
   - Evaluates runs prevented relative to the ML Expected Runs model ($xRuns$) and extreme pressure suppression.

### Dual-Role / All-Rounder Handling
Players who meet minimum thresholds in both batting and bowling ($\ge 15$ balls in both disciplines) are flagged as **Verified All-Rounders** (e.g. Ravindra Jadeja, Rashid Khan, Dwayne Bravo). The dashboard renders **two independent radar charts** side-by-side, never blending unrelated batting and bowling metrics into a misleading single polygon.

---

## 4. Key Interactive Modules

1. **👤 Player Scouting Profile:**
   - Single-player deep dive with interactive 5-axis radar chart, progress breakdown, KPI panels, and sample confidence flags.
2. **⚔️ Head-to-Head Comparison:**
   - Multi-player comparison (select up to 4 players simultaneously) with overlaid polygons in distinct neon colors, differential dimension bar charts, and side-by-side statistical cards.
3. **⚡ Pressure vs Normal Context:**
   - Interactive scatter quadrants analyzing $SR_{Normal}$ vs $SR_{Pressure}$ (Batting) and $Eco_{Normal}$ vs $Eco_{Pressure}$ (Bowling) with parity reference diagonals ($y=x$).
4. **🌍 Competition Dynamics:**
   - Drill-down across 7 leagues (IPL, T20I, BBL, SA20, MLC, ILT20, LPL) and 5 seasons (2022–2026).
5. **🤖 GPU Machine Learning Intelligence:**
   - Visualizes model performance benchmarks (RMSE, Log-Loss, Brier Score), SHAP feature gain importances, and expected runs calibration.
6. **🏆 Master Leaderboards & 19 Roster:**
   - Filterable tables for the entire dataset plus the locked 19-player challenge roster with verified justifications.

---

## 5. How to Launch the Dashboard Locally

### Option 1: Direct Python Execution
```bash
# Navigate to the dashboard directory
cd Pressure_Performance_Intelligence/Supporting_Files/Dashboard

# Install required dependencies
pip install -r requirements.txt

# Run application
python app.py
```
Open your browser and navigate to **`http://127.0.0.1:8050`**.

### Option 2: Windows Batch Script (One-Click)
Double-click:
`Pressure_Performance_Intelligence\Supporting_Files\Dashboard\run_dashboard.bat`

### Option 3: PowerShell Script
```powershell
.\Pressure_Performance_Intelligence\Supporting_Files\Dashboard\run_dashboard.ps1
```
