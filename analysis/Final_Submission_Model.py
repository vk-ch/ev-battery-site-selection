"""
SAMSUNG $5B BATTERY PLANT - FINAL SUBMISSION MODEL
ML + Domain Expert Framework with Multi-Dimensional Analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("SAMSUNG EV BATTERY PLANT - FINAL SUBMISSION MODEL")
print("="*80)
print("Framework: ML Pattern Discovery + Domain Expert (4 Pillars)")
print("="*80)

# ============================================================================
# 1. LOAD & PREPARE DATA
# ============================================================================
print("\n[1/9] Loading data...")

df_tax = pd.read_csv('df_tax_data.csv')
df_factories = pd.read_csv('us_ev_battery_factories.csv')

# Target variable
factory_counts = df_factories[
    df_factories['status'].isin(['Operational', 'Under Construction'])
].groupby('state').size().reset_index(name='target')

# ============================================================================
# 2. ENGINEER ML FEATURES (Keep Same)
# ============================================================================
print("[2/9] Engineering ML features...")

# Tax features with CAGR
tax_mapping = {
    'T09': 'sales_tax',
    'T13': 'gas_tax',
    'T15': 'utilities_tax',
    'T40': 'income_tax',
    'T41': 'corporate_tax',
    'T01': 'property_tax',
}

def calc_cagr(group):
    years = sorted(group['Year'].unique())
    if len(years) < 2:
        return 0
    start = group[group['Year'] == years[0]]['Amount'].sum()
    end = group[group['Year'] == years[-1]]['Amount'].sum()
    if start <= 0 or end <= 0:
        return 0
    return ((end / start) ** (1 / (len(years) - 1)) - 1) * 100

df_tax_filtered = df_tax[df_tax['Year'] >= 2015].copy()

tax_features = []
for code, name in tax_mapping.items():
    df_code = df_tax_filtered[df_tax_filtered['Tax_Code'] == code]
    cagr = df_code.groupby('State').apply(calc_cagr).reset_index(name=f'{name}_cagr')
    amount = df_code[df_code['Year'] >= 2020].groupby('State')['Amount'].mean().reset_index(
        name=f'{name}_amount'
    )
    merged = cagr.merge(amount, on='State', how='outer')
    tax_features.append(merged)

tax_df = tax_features[0]
for df in tax_features[1:]:
    tax_df = tax_df.merge(df, on='State', how='outer')
tax_df.columns = ['state'] + list(tax_df.columns[1:])

# Economic features
econ_cols = [
    'State', 'Population', 'Unemployment_Rate', 'Personal_Income',
    'GDP_Manufacturing', 'GDP_Transport_Warehousing', 'GDP_Utilities',
    'GDP_Total', 'GDP_Durable_Goods', 'Total Civilian Labor Force',
    'Total Employed Civilian Labor Force'
]
df_econ = df_tax[df_tax['Year'] >= 2020][econ_cols].groupby('State').mean().reset_index()
df_econ.columns = ['state'] + [col.lower().replace(' ', '_') for col in df_econ.columns[1:]]

# ============================================================================
# 3. ENGINEER DOMAIN EXPERT FEATURES (NEW)
# ============================================================================
print("[3/9] Engineering domain expert features (4 pillars)...")

# PILLAR 1: Tax Environment
print("   - Tax Environment features...")

# Tax per capita (total tax revenue / population)
total_tax = df_tax[df_tax['Year'] >= 2020].groupby(['State', 'Year'])['Amount'].sum().reset_index()
total_tax_avg = total_tax.groupby('State')['Amount'].mean().reset_index(name='total_tax_revenue')

pop_avg = df_tax[df_tax['Year'] >= 2020].groupby('State')['Population'].mean().reset_index()
tax_percapita = total_tax_avg.merge(pop_avg, on='State')
tax_percapita['tax_per_capita'] = (tax_percapita['total_tax_revenue'] / tax_percapita['Population']) * 1000
tax_percapita = tax_percapita[['State', 'tax_per_capita', 'total_tax_revenue']].rename(columns={'State': 'state'})

# Tax revenue volatility (coefficient of variation)
total_tax_volatility = total_tax.groupby('State')['Amount'].agg(['mean', 'std']).reset_index()
total_tax_volatility['tax_cv'] = (total_tax_volatility['std'] / total_tax_volatility['mean']) * 100
total_tax_volatility = total_tax_volatility[['State', 'tax_cv']].rename(columns={'State': 'state'})

# PILLAR 2: Labor Market Strength
print("   - Labor Market Strength features...")

# Labor force participation rate
df_econ['labor_force_participation'] = (
    df_econ['total_employed_civilian_labor_force'] / df_econ['total_civilian_labor_force']
) * 100

# Unemployment volatility
unemp_volatility = df_tax[df_tax['Year'] >= 2020].groupby('State')['Unemployment_Rate'].agg(['mean', 'std']).reset_index()
unemp_volatility['unemployment_cv'] = (unemp_volatility['std'] / unemp_volatility['mean']) * 100
unemp_volatility = unemp_volatility[['State', 'unemployment_cv']].rename(columns={'State': 'state'})

# PILLAR 3: Industrial & Infrastructure Base
print("   - Industrial & Infrastructure features...")

# Manufacturing share of GDP
df_econ['manufacturing_share_gdp'] = (df_econ['gdp_manufacturing'] / df_econ['gdp_total']) * 100

# PILLAR 4: Business Dynamism
print("   - Business Dynamism features...")

# Business formations (use Applications as proxy)
business_df = df_tax[df_tax['Year'] >= 2020].groupby('State')['Applications'].mean().reset_index()
business_df.columns = ['state', 'business_formations']

# Business formations per capita
business_percapita = business_df.merge(pop_avg.rename(columns={'State': 'state'}), on='state')
business_percapita['business_formations_percapita'] = (
    business_percapita['business_formations'] / business_percapita['Population']
) * 100000  # Per 100k people

# ============================================================================
# 4. MERGE ALL FEATURES
# ============================================================================
print("[4/9] Merging all features...")

features_df = df_econ.merge(tax_df, on='state', how='outer')
features_df = features_df.merge(tax_percapita, on='state', how='outer')
features_df = features_df.merge(total_tax_volatility, on='state', how='outer')
features_df = features_df.merge(unemp_volatility, on='state', how='outer')
features_df = features_df.merge(business_percapita[['state', 'business_formations', 'business_formations_percapita']], on='state', how='outer')

# Add target
all_states = df_tax['State'].unique()
y_all = pd.DataFrame({'state': all_states})
y_all = y_all.merge(factory_counts, on='state', how='left').fillna(0)

final_df = features_df.merge(y_all, on='state', how='inner')
final_df = final_df[final_df['state'] != 'US Total']
final_df = final_df.fillna(0).replace([np.inf, -np.inf], 0)

print(f"✓ {len(final_df)} states, {final_df.shape[1]-2} features")

# ============================================================================
# 5. NORMALIZE FEATURES
# ============================================================================
print("[5/9] Normalizing features...")

def normalize_positive(series, cap_percentile=95):
    min_val = series.min()
    max_val = series.quantile(cap_percentile / 100)
    normalized = (series.clip(upper=max_val) - min_val) / (max_val - min_val) * 100
    return normalized.fillna(0)

def normalize_negative(series, cap_percentile=95):
    min_val = series.min()
    max_val = series.quantile(cap_percentile / 100)
    normalized = (max_val - series.clip(upper=max_val)) / (max_val - min_val) * 100
    return normalized.fillna(0)

def normalize_optimal_range(series, optimal_min, optimal_max):
    scores = []
    for val in series:
        if optimal_min <= val <= optimal_max:
            score = 100
        elif val < optimal_min:
            distance = optimal_min - val
            score = max(0, 100 - distance * 10)
        else:
            distance = val - optimal_max
            score = max(0, 100 - distance * 10)
        scores.append(score)
    return pd.Series(scores, index=series.index)

def normalize_stability(series):
    abs_series = series.abs()
    max_val = abs_series.quantile(0.95)
    normalized = (max_val - abs_series.clip(upper=max_val)) / max_val * 100
    return normalized.fillna(0)

norm_df = pd.DataFrame({'state': final_df['state']})

# ML Features (normalized)
norm_df['gdp_manufacturing_norm'] = normalize_positive(final_df['gdp_manufacturing'])
norm_df['gdp_utilities_norm'] = normalize_positive(final_df['gdp_utilities'])
norm_df['gdp_transport_warehousing_norm'] = normalize_positive(final_df['gdp_transport_warehousing'])
norm_df['population_norm'] = normalize_positive(final_df['population'])
norm_df['gas_tax_cagr_norm'] = normalize_optimal_range(final_df['gas_tax_cagr'], 0, 5)
norm_df['income_tax_cagr_norm'] = normalize_stability(final_df['income_tax_cagr'])
norm_df['corporate_tax_cagr_norm'] = normalize_stability(final_df['corporate_tax_cagr'])
norm_df['utilities_tax_amount_norm'] = normalize_negative(final_df['utilities_tax_amount'])
norm_df['unemployment_norm'] = normalize_optimal_range(final_df['unemployment_rate'], 3, 5)
norm_df['sales_tax_amount_norm'] = normalize_negative(final_df['sales_tax_amount'])

# Domain Expert Features (normalized)
# Pillar 1: Tax Environment
norm_df['tax_per_capita_norm'] = normalize_negative(final_df['tax_per_capita'])
norm_df['tax_cv_norm'] = normalize_negative(final_df['tax_cv'])
norm_df['total_tax_revenue_norm'] = normalize_positive(final_df['total_tax_revenue'])

# Pillar 2: Labor Market
norm_df['labor_force_norm'] = normalize_positive(final_df['total_civilian_labor_force'])
norm_df['unemployment_rate_norm'] = normalize_negative(final_df['unemployment_rate'])
norm_df['labor_participation_norm'] = normalize_positive(final_df['labor_force_participation'])
norm_df['unemployment_cv_norm'] = normalize_negative(final_df['unemployment_cv'])

# Pillar 3: Industrial Base
norm_df['manufacturing_share_norm'] = normalize_positive(final_df['manufacturing_share_gdp'])
norm_df['gdp_total_norm'] = normalize_positive(final_df['gdp_total'])
norm_df['gdp_durable_goods_norm'] = normalize_positive(final_df['gdp_durable_goods'])

# Pillar 4: Business Dynamism
norm_df['business_formations_norm'] = normalize_positive(final_df['business_formations'])
norm_df['business_percapita_norm'] = normalize_positive(final_df['business_formations_percapita'])

# Bonuses
norm_df['no_income_tax_bonus'] = (final_df['income_tax_amount'] == 0).astype(int) * 100
norm_df['no_corporate_tax_bonus'] = (final_df['corporate_tax_amount'] == 0).astype(int) * 100

norm_df = norm_df.merge(final_df[['state', 'target']], on='state')
norm_df.rename(columns={'target': 'current_factories'}, inplace=True)

print(f"✓ {len(norm_df.columns)-2} normalized features")

# ============================================================================
# 6. DEFINE SCORING MODELS
# ============================================================================
print("[6/9] Defining scoring models...")

# ML Weights (60% of hybrid)
ml_weights = {
    'gas_tax_cagr_norm': 0.14,
    'income_tax_cagr_norm': 0.116,
    'gdp_manufacturing_norm': 0.111,
    'utilities_tax_amount_norm': 0.11,
    'corporate_tax_cagr_norm': 0.089,
    'population_norm': 0.10,
    'gdp_transport_warehousing_norm': 0.08,
    'gdp_utilities_norm': 0.075,
    'unemployment_norm': 0.05,
    'sales_tax_amount_norm': 0.04,
    'no_income_tax_bonus': 0.03,
    'no_corporate_tax_bonus': 0.03,
}

# Domain Expert Weights (40% of hybrid) - 4 Pillars
# Pillar 1: Tax Environment (25% of domain = 10% of total)
tax_env_weights = {
    'tax_per_capita_norm': 0.06,
    'tax_cv_norm': 0.03,
    'total_tax_revenue_norm': 0.01,
}

# Pillar 2: Labor Market Strength (30% of domain = 12% of total)
labor_weights = {
    'labor_force_norm': 0.06,
    'unemployment_rate_norm': 0.03,
    'labor_participation_norm': 0.02,
    'unemployment_cv_norm': 0.01,
}

# Pillar 3: Industrial & Infrastructure Base (35% of domain = 14% of total)
industrial_weights = {
    'manufacturing_share_norm': 0.06,
    'gdp_manufacturing_norm': 0.04,
    'gdp_utilities_norm': 0.02,
    'gdp_transport_warehousing_norm': 0.01,
    'gdp_durable_goods_norm': 0.01,
}

# Pillar 4: Business Dynamism (10% of domain = 4% of total)
business_weights = {
    'business_formations_norm': 0.02,
    'business_percapita_norm': 0.02,
}

domain_weights = {**tax_env_weights, **labor_weights, **industrial_weights, **business_weights}

# Hybrid: 60% ML + 40% Domain
all_features = set(list(ml_weights.keys()) + list(domain_weights.keys()))
hybrid_weights = {}
for feat in all_features:
    ml_w = ml_weights.get(feat, 0)
    domain_w = domain_weights.get(feat, 0)
    hybrid_weights[feat] = (ml_w * 0.6) + (domain_w * 0.4)

total = sum(hybrid_weights.values())
hybrid_weights = {k: v/total for k, v in hybrid_weights.items()}

print(f"✓ ML: {len(ml_weights)} features")
print(f"✓ Domain: {len(domain_weights)} features (4 pillars)")
print(f"✓ Hybrid: {len(hybrid_weights)} features (60/40 split)")

# ============================================================================
# 7. CALCULATE SCORES (Multiple Dimensions)
# ============================================================================
print("[7/9] Calculating multi-dimensional scores...")

def calculate_score(norm_df, weights):
    scores = []
    for idx, row in norm_df.iterrows():
        score = sum(row.get(feat, 0) * weight for feat, weight in weights.items())
        scores.append(score)
    return scores

# Overall scores
norm_df['ml_score'] = calculate_score(norm_df, ml_weights)
norm_df['domain_score'] = calculate_score(norm_df, domain_weights)
norm_df['hybrid_score'] = calculate_score(norm_df, hybrid_weights)

# Sub-dimension scores
norm_df['tax_environment_score'] = calculate_score(norm_df, tax_env_weights)
norm_df['labor_market_score'] = calculate_score(norm_df, labor_weights)
norm_df['industrial_base_score'] = calculate_score(norm_df, industrial_weights)
norm_df['business_dynamism_score'] = calculate_score(norm_df, business_weights)

# Composite dimensions for analysis
norm_df['tax_burden_score'] = norm_df['tax_per_capita_norm']  # Lower tax = higher score
norm_df['stability_score'] = (
    norm_df['tax_cv_norm'] * 0.5 + norm_df['unemployment_cv_norm'] * 0.5
)
norm_df['policy_attractiveness_score'] = (
    norm_df['no_income_tax_bonus'] * 0.4 +
    norm_df['no_corporate_tax_bonus'] * 0.4 +
    norm_df['business_percapita_norm'] * 0.2
)

print("✓ Overall scores calculated")
print("✓ Sub-dimension scores calculated")
print("✓ Composite scores calculated")

# ============================================================================
# 8. RANK STATES
# ============================================================================
print("[8/9] Ranking states...")

rankings = norm_df[[
    'state', 'hybrid_score', 'ml_score', 'domain_score',
    'tax_environment_score', 'labor_market_score', 'industrial_base_score',
    'business_dynamism_score', 'tax_burden_score', 'stability_score',
    'policy_attractiveness_score', 'current_factories'
]].copy()

rankings = rankings.sort_values('hybrid_score', ascending=False).reset_index(drop=True)
rankings['rank'] = range(1, len(rankings) + 1)

def assign_tier(score):
    if score >= 20:
        return 'Tier 1'
    elif score >= 15:
        return 'Tier 2'
    elif score >= 10:
        return 'Tier 3'
    else:
        return 'Tier 4'

rankings['tier'] = rankings['hybrid_score'].apply(assign_tier)

# ============================================================================
# 9. GENERATE OUTPUTS
# ============================================================================
print("[9/9] Generating outputs...")

output_dir = '/sessions/brave-intelligent-knuth/mnt/outputs/'

# Output 1: Overall Rankings
rankings.to_csv(output_dir + 'final_submission_overall_rankings.csv', index=False)
print("✓ Output 1: Overall rankings saved")

# Output 2: Detailed scoring breakdown
scoring_breakdown = rankings[rankings['rank'] <= 20][[
    'rank', 'state', 'hybrid_score',
    'tax_environment_score', 'labor_market_score',
    'industrial_base_score', 'business_dynamism_score',
    'tax_burden_score', 'stability_score', 'policy_attractiveness_score'
]]
scoring_breakdown.to_csv(output_dir + 'final_submission_scoring_breakdown.csv', index=False)
print("✓ Output 2: Scoring breakdown (top 20) saved")

# Output 3: Create tradeoff bubble charts
print("✓ Output 3: Creating visualizations...")

plt.style.use('seaborn-v0_8-darkgrid')
fig = plt.figure(figsize=(20, 12))

# Chart 1: Tax Burden vs Industrial Base (bubble = labor market)
ax1 = plt.subplot(2, 3, 1)
top20 = rankings.head(20)
scatter1 = ax1.scatter(
    top20['tax_burden_score'],
    top20['industrial_base_score'],
    s=top20['labor_market_score'] * 50,
    c=top20['hybrid_score'],
    cmap='RdYlGn',
    alpha=0.6,
    edgecolors='black',
    linewidth=1.5
)

for idx, row in top20.iterrows():
    if row['rank'] <= 10:
        ax1.annotate(
            row['state'],
            (row['tax_burden_score'], row['industrial_base_score']),
            fontsize=8,
            ha='center'
        )

ax1.set_xlabel('Tax Burden Score (Lower tax = Higher)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Industrial Base Score', fontsize=11, fontweight='bold')
ax1.set_title('Tradeoff: Tax Burden vs Industrial Base\n(Bubble size = Labor Market)',
              fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
plt.colorbar(scatter1, ax=ax1, label='Overall Score')

# Chart 2: Stability vs Policy Attractiveness
ax2 = plt.subplot(2, 3, 2)
scatter2 = ax2.scatter(
    top20['stability_score'],
    top20['policy_attractiveness_score'],
    s=top20['business_dynamism_score'] * 100,
    c=top20['hybrid_score'],
    cmap='RdYlGn',
    alpha=0.6,
    edgecolors='black',
    linewidth=1.5
)

for idx, row in top20.iterrows():
    if row['rank'] <= 10:
        ax2.annotate(
            row['state'],
            (row['stability_score'], row['policy_attractiveness_score']),
            fontsize=8,
            ha='center'
        )

ax2.set_xlabel('Stability Score (Tax + Labor)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Policy Attractiveness Score', fontsize=11, fontweight='bold')
ax2.set_title('Tradeoff: Stability vs Policy Attractiveness\n(Bubble size = Business Dynamism)',
              fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
plt.colorbar(scatter2, ax=ax2, label='Overall Score')

# Chart 3: Tax Environment vs Labor Market
ax3 = plt.subplot(2, 3, 3)
scatter3 = ax3.scatter(
    top20['tax_environment_score'],
    top20['labor_market_score'],
    s=top20['industrial_base_score'] * 50,
    c=top20['hybrid_score'],
    cmap='RdYlGn',
    alpha=0.6,
    edgecolors='black',
    linewidth=1.5
)

for idx, row in top20.iterrows():
    if row['rank'] <= 10:
        ax3.annotate(
            row['state'],
            (row['tax_environment_score'], row['labor_market_score']),
            fontsize=8,
            ha='center'
        )

ax3.set_xlabel('Tax Environment Score', fontsize=11, fontweight='bold')
ax3.set_ylabel('Labor Market Score', fontsize=11, fontweight='bold')
ax3.set_title('Tradeoff: Tax Environment vs Labor Market\n(Bubble size = Industrial Base)',
              fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)
plt.colorbar(scatter3, ax=ax3, label='Overall Score')

# Chart 4: 4-Pillar Radar Chart (Top 5 states)
ax4 = plt.subplot(2, 3, 4, projection='polar')
top5 = rankings.head(5)

categories = ['Tax\nEnvironment', 'Labor\nMarket', 'Industrial\nBase', 'Business\nDynamism']
angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
angles += angles[:1]

colors = ['#1E2761', '#1428A0', '#28A745', '#FF9500', '#DC3545']

for idx, (_, row) in enumerate(top5.iterrows()):
    values = [
        row['tax_environment_score'],
        row['labor_market_score'],
        row['industrial_base_score'],
        row['business_dynamism_score']
    ]
    values += values[:1]

    ax4.plot(angles, values, 'o-', linewidth=2, label=row['state'], color=colors[idx])
    ax4.fill(angles, values, alpha=0.15, color=colors[idx])

ax4.set_xticks(angles[:-1])
ax4.set_xticklabels(categories, fontsize=10, fontweight='bold')
ax4.set_ylim(0, max(top5['industrial_base_score'].max(), top5['labor_market_score'].max()) * 1.1)
ax4.set_title('Top 5 States: 4-Pillar Comparison', fontsize=12, fontweight='bold', pad=20)
ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
ax4.grid(True)

# Chart 5: Stacked bar for top 10 states
ax5 = plt.subplot(2, 3, 5)
top10 = rankings.head(10)

x = np.arange(len(top10))
width = 0.8

p1 = ax5.bar(x, top10['tax_environment_score'], width, label='Tax Environment', color='#1E2761')
p2 = ax5.bar(x, top10['labor_market_score'], width, bottom=top10['tax_environment_score'],
             label='Labor Market', color='#1428A0')
p3 = ax5.bar(x, top10['industrial_base_score'], width,
             bottom=top10['tax_environment_score'] + top10['labor_market_score'],
             label='Industrial Base', color='#28A745')
p4 = ax5.bar(x, top10['business_dynamism_score'], width,
             bottom=top10['tax_environment_score'] + top10['labor_market_score'] + top10['industrial_base_score'],
             label='Business Dynamism', color='#FF9500')

ax5.set_xlabel('State', fontsize=11, fontweight='bold')
ax5.set_ylabel('Score', fontsize=11, fontweight='bold')
ax5.set_title('Top 10 States: Score Composition by Pillar', fontsize=12, fontweight='bold')
ax5.set_xticks(x)
ax5.set_xticklabels(top10['state'], rotation=45, ha='right', fontsize=9)
ax5.legend(loc='upper right', fontsize=9)
ax5.grid(True, alpha=0.3, axis='y')

# Chart 6: Final Top 3-4 Comparison
ax6 = plt.subplot(2, 3, 6)
top4 = rankings.head(4)

metrics = ['Tax\nEnvironment', 'Labor\nMarket', 'Industrial\nBase', 'Business\nDynamism',
           'Tax\nBurden', 'Stability', 'Policy\nAttractiveness']
x_pos = np.arange(len(metrics))
bar_width = 0.2

for idx, (_, row) in enumerate(top4.iterrows()):
    values = [
        row['tax_environment_score'],
        row['labor_market_score'],
        row['industrial_base_score'],
        row['business_dynamism_score'],
        row['tax_burden_score'],
        row['stability_score'],
        row['policy_attractiveness_score']
    ]

    ax6.bar(x_pos + idx * bar_width, values, bar_width, label=f"#{idx+1} {row['state']}",
            color=colors[idx])

ax6.set_xlabel('Metric', fontsize=11, fontweight='bold')
ax6.set_ylabel('Score', fontsize=11, fontweight='bold')
ax6.set_title('Final Top 4 States: Comprehensive Comparison', fontsize=12, fontweight='bold')
ax6.set_xticks(x_pos + bar_width * 1.5)
ax6.set_xticklabels(metrics, fontsize=9, rotation=45, ha='right')
ax6.legend(fontsize=9, loc='upper right')
ax6.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(output_dir + 'final_submission_visualizations.png', dpi=300, bbox_inches='tight')
plt.close()

print("✓ Output 3: Visualizations saved")

# Output 4: Summary statistics
print("\n" + "="*80)
print("FINAL SUBMISSION - TOP 10 STATES")
print("="*80)

summary = rankings.head(10)[['rank', 'state', 'hybrid_score', 'tier', 'current_factories']]
print("\n" + summary.to_string(index=False))

print("\n" + "="*80)
print("TOP 4 STATES - DETAILED BREAKDOWN")
print("="*80)

for idx in range(4):
    row = rankings.iloc[idx]
    print(f"\n{'='*80}")
    print(f"#{row['rank']}: {row['state'].upper()} (Score: {row['hybrid_score']:.2f})")
    print(f"{'='*80}")
    print(f"  Tax Environment:        {row['tax_environment_score']:.2f}/100")
    print(f"  Labor Market:           {row['labor_market_score']:.2f}/100")
    print(f"  Industrial Base:        {row['industrial_base_score']:.2f}/100")
    print(f"  Business Dynamism:      {row['business_dynamism_score']:.2f}/100")
    print(f"  Tax Burden (low=good):  {row['tax_burden_score']:.2f}/100")
    print(f"  Stability:              {row['stability_score']:.2f}/100")
    print(f"  Policy Attractiveness:  {row['policy_attractiveness_score']:.2f}/100")
    print(f"  Current Factories:      {int(row['current_factories'])}")

print("\n" + "="*80)
print("EXPORTS COMPLETED")
print("="*80)
print(f"✓ final_submission_overall_rankings.csv")
print(f"✓ final_submission_scoring_breakdown.csv")
print(f"✓ final_submission_visualizations.png")

print("\n" + "="*80)
print("🏆 FINAL RECOMMENDATION")
print("="*80)

winner = rankings.iloc[0]
print(f"""
WINNER: {winner['state'].upper()}
Overall Score: {winner['hybrid_score']:.2f}/100
Tier: {winner['tier']}

{winner['state']} ranks #1 due to superior performance across all 4 pillars:
- Tax Environment: {winner['tax_environment_score']:.1f}/100
- Labor Market: {winner['labor_market_score']:.1f}/100
- Industrial Base: {winner['industrial_base_score']:.1f}/100
- Business Dynamism: {winner['business_dynamism_score']:.1f}/100

Recommended Action: Proceed with site selection in {winner['state']}.
""")

print("="*80)
print("ANALYSIS COMPLETE")
print("="*80)
